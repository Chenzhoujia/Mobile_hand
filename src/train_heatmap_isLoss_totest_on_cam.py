# -*- coding: utf-8 -*-
import datetime

import scipy.misc
import tensorflow as tf
import os
import platform
import time
import numpy as np
import configparser
import cv2
from tqdm import tqdm

from dataset_interface.RHD import RHD
from dataset_interface.dataset_prepare import CocoPose
from src.networks import get_network
import matplotlib.pyplot as plt
from src.general import NetworkOps
from src import  network_mv2_hourglass
from mpl_toolkits.mplot3d import Axes3D
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
ops = NetworkOps
test_num = 1
def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor],
                                    name=name)
def get_loss_and_output(model, batchsize, input_image, scoremap, is_loss, reuse_variables=None):
    losses = []

    # 叠加在batch上重用特征提取网络
    input_image = tf.add(input_image, 0, name='input_image')
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        network_mv2_hourglass.N_KPOINTS = 2
        _, pred_heatmaps_all = get_network(model, input_image, True) #第一个batch的维度 hand back

    loss_scoremap = 0.0
    loss_is_loss = 0.0
    for loss_i in range(len(pred_heatmaps_all)):
        # 计算 isloss，用softmax计算 0~1}
        is_loss_s = pred_heatmaps_all[loss_i].get_shape().as_list()
        pre_is_loss = tf.reshape(pred_heatmaps_all[loss_i], [-1, is_loss_s[1]*is_loss_s[2]*is_loss_s[3]])  # this is Bx16*16*1
        out_chan_list = [32, 16, 8, 2]
        for i, out_chan in enumerate(out_chan_list):
            pre_is_loss = ops.fully_connected_relu(pre_is_loss, 'is_loss_fc_%d_%d' % (loss_i, i), out_chan=out_chan, trainable=True)#(128,1)

        # 计算热度图
        scale = 2
        pred_heatmaps_tmp = upsample(pred_heatmaps_all[loss_i], scale, name="upsample_for_hotmap_loss_%d" % loss_i)


        #用is loss 修正热度图
        pre_is_loss = tf.nn.softmax(pre_is_loss)
        pred_heatmaps_tmp_01_modi = tf.expand_dims(tf.expand_dims(pre_is_loss, axis=1), axis=1)*pred_heatmaps_tmp
        pred_heatmaps_tmp = tf.nn.softmax(pred_heatmaps_tmp)
        pred_heatmaps_tmp_01_modi = tf.nn.softmax(pred_heatmaps_tmp_01_modi)

    total_loss = loss_scoremap + loss_is_loss
    return pred_heatmaps_tmp, pre_is_loss, pred_heatmaps_tmp_01_modi

def average_gradients(tower_grads):
    """
    Get gradients of all variables.
    :param tower_grads:
    :return:
    """
    average_grads = []

    # get variable and gradients in differents gpus
    for grad_and_vars in zip(*tower_grads):
        # calculate the average gradient of each gpu
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main(argv=None):
    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config_file = "../experiments/mv2_cpm.cfg"
    if len(argv) != 1:
        config_file = argv[1]
    config.read(config_file)
    for _ in config.options("Train"):
        params[_] = eval(config.get("Train", _))

    os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']

    gpus_index = params['visible_devices'].split(",")
    params['gpus'] = len(gpus_index)

    if not os.path.exists(params['modelpath']):
        os.makedirs(params['modelpath'])
    if not os.path.exists(params['logpath']):
        os.makedirs(params['logpath'])

    gpus = 'gpus'
    if platform.system() == 'Darwin':
        gpus = 'cpu'
    training_name = '{}_batch-{}_lr-{}_{}-{}_{}x{}_{}'.format(
        params['model'],
        params['batchsize'],
        params['lr'],
        gpus,
        params['gpus'],
        params['input_width'], params['input_height'],
        config_file.replace("/", "-").replace(".cfg", "")
    )

    with tf.Graph().as_default(), tf.device("/cpu:0"):
        dataset_RHD = RHD(batchnum=test_num)

        global_step = tf.Variable(0, trainable=False)

        reuse_variable = False

        for i in range(params['gpus']):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("GPU_%d" % i):
                    input_node = tf.placeholder(tf.float32, shape=[test_num, 32, 32, 3], name="input_image")

                    batch_data_all = dataset_RHD.get_batch_data
                    input_image1 = batch_data_all[8]
                    input_image2 = batch_data_all[10]
                    hand_motion = batch_data_all[9]
                    scoremap1 = batch_data_all[11]
                    scoremap2 = batch_data_all[12]
                    is_loss1 = batch_data_all[13]
                    is_loss2 = batch_data_all[14]

                    # input_image = tf.concat([input_image1, input_image1_back], 0) #第一个batch的维度 hand1 back1
                    # scoremap = tf.concat([scoremap1, scoremap1_back], 0)
                    # is_loss = tf.concat([is_loss1, is_loss1_back], 0)

                    input_image = input_node
                    scoremap = scoremap1
                    is_loss = is_loss1


                    # 计算一个scoremap的loss
                    scoremap = tf.reduce_sum(scoremap, axis=-1)
                    one_scoremap = tf.ones_like(scoremap)
                    scoremap = tf.where(scoremap > 1, x=one_scoremap, y=scoremap)
                    scoremap = tf.expand_dims(scoremap, axis=-1)  # hand back
                    is_loss = tf.expand_dims(is_loss, axis=-1)

                    """
                    model, batchsize, input_image, scoremap, is_loss, reuse_variables=None
                    total_loss, loss_is_loss, loss_scoremap, pred_heatmaps_tmp, pre_is_loss, pred_heatmaps_tmp_01_modi
                    """
                    preheat, pre_is_loss, pred_heatmaps_tmp_01_modi\
                        = get_loss_and_output(params['model'], params['batchsize'],
                                                input_image, scoremap, is_loss, reuse_variable)


        saver = tf.train.Saver(max_to_keep=10)


        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            cap = cv2.VideoCapture(0)
            centerx = 160
            centery = 120
            size = 24
            color_ = (0, 255, 0)
            checkpoint_path = os.path.join(params['modelpath'], training_name)
            model_name = 'model-4200'
            if checkpoint_path:
                saver.restore(sess, checkpoint_path+'/'+model_name)
                print("restore from " + checkpoint_path+'/'+model_name)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize']* 2 * params['gpus'])

            print("Start testing...")
            path = "/home/chen/Documents/Mobile_hand/experiments/varify/image/set1/"
            import matplotlib.image
            for step in tqdm(range(100000)):
                _, image_raw1 = cap.read()
                image_raw1 = scipy.misc.imresize(image_raw1, (240, 320))
                image_raw1_crop = np.array(image_raw1[centery - size: centery + size, centerx - size: centerx + size])
                image_raw1_crop = cv2.resize(image_raw1_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)
                first_point = (centerx-size, centery-size)
                last_point = (centerx+size, centery+size)
                cv2.rectangle(image_raw1, first_point, last_point, color=color_, thickness = 2)

                image_raw12_crop = image_raw1_crop.astype('float') / 255.0 - 0.5
                scoremap_v, is_loss_v,\
                preheat_v, pre_is_loss_v, pred_heatmaps_tmp_01_modi_v\
                    = sess.run(
                    [scoremap, is_loss,
                     preheat, pre_is_loss, pred_heatmaps_tmp_01_modi],
                    feed_dict={input_node: np.repeat(image_raw12_crop[np.newaxis, :],test_num,axis=0)})


                #根据preheat_v 计算最有可能的指尖坐标，当手指指尖存在时更新坐标centerx， centery
                if pre_is_loss_v[0, 0]>pre_is_loss_v[0, 1]:
                    color_ = (0, 255, 0)
                    motion = preheat_v[0, :, :, 0] - preheat_v[0, :, :, 1]
                    raw, column = motion.shape
                    _positon = np.argmax(motion)  # get the index of max in the a
                    m, n = divmod(_positon, column)
                else:
                    color_ = (255, 0, 0)
                    m = 15.5
                    n = 15.5

                right_move = int((n - 15.5)/32*48)
                down_move = int((m - 15.5)/32*48)
                centery = centery+down_move
                centerx = centerx+right_move
                input_image_v = (image_raw12_crop + 0.5) * 255
                input_image_v = input_image_v.astype(np.int16)

                if centery<0 or centery>240:
                    centery = 120

                if centerx < 0 or centerx > 320:
                    centerx = 160


                fig = plt.figure(1)
                plt.clf()
                ax1 = fig.add_subplot(2, 3, 1)
                ax1.imshow(input_image_v)  # 第一个batch的维度 hand1(0~31) back1(32~63)

                ax3 = fig.add_subplot(2, 3, 2)
                ax3.imshow(preheat_v[0, :, :, 0])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                ax3.set_title(str(pre_is_loss_v[0, 0]))  # hand1 back1

                ax7 = fig.add_subplot(2, 3, 5)
                ax7.imshow(preheat_v[0, :, :, 1])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                ax7.set_title(str(pre_is_loss_v[0, 1]))  # hand1 back1

                ax4 = fig.add_subplot(2, 3, 3)
                ax4.imshow(pred_heatmaps_tmp_01_modi_v[0, :, :, 0])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                ax4.set_title('m:'+str(m)+' n:'+str(n))

                ax8 = fig.add_subplot(2, 3, 6)
                ax8.imshow(pred_heatmaps_tmp_01_modi_v[0, :, :, 1])  # 第一个batch的维度 hand1(0~31) back1(32~63)

                ax2 = fig.add_subplot(2, 3, 4)
                ax2.imshow(image_raw1)  # 第一个batch的维度 hand1(0~31) back1(32~63)

                plt.savefig("/home/chen/Documents/Mobile_hand/experiments/varify/image/valid_on_cam/softmax/"+ str(step).zfill(10) + model_name+"_.png")
                plt.pause(0.01)

if __name__ == '__main__':
    tf.app.run()
