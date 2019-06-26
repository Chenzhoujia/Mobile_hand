# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import platform
import time
import numpy as np
import configparser

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
    loss_scoremap_m = 0.0
    loss_is_loss = 0.0
    for loss_i in range(len(pred_heatmaps_all)):
        # 计算 isloss，用softmax计算 0~1}
        is_loss_s = pred_heatmaps_all[loss_i].get_shape().as_list()
        pre_is_loss = tf.reshape(pred_heatmaps_all[loss_i], [-1, is_loss_s[1]*is_loss_s[2]*is_loss_s[3]])  # this is Bx16*16*1
        out_chan_list = [32, 16, 8, 2]
        for i, out_chan in enumerate(out_chan_list):
            pre_is_loss = ops.fully_connected_relu(pre_is_loss, 'is_loss_fc_%d_%d' % (loss_i, i), out_chan=out_chan, trainable=True)#(128,1)
        #将pre_is_loss [?,2]
        loss_is_loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pre_is_loss, labels=is_loss))

        # 计算热度图
        scale = 2
        pred_heatmaps_tmp = upsample(pred_heatmaps_all[loss_i], scale, name="upsample_for_hotmap_loss_%d" % loss_i)

        #在计算loss时将其约束在01之间可以增加估计热度图的对比度
        s = scoremap.get_shape().as_list()
        gt = tf.reshape(scoremap, [batchsize * s[1] * s[2], -1])
        pred = tf.reshape(pred_heatmaps_tmp, [batchsize * s[1] * s[2], -1])
        loss_scoremap += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=gt))

        # one_tmp = tf.ones_like(pred_heatmaps_tmp)
        # zero_tmp = tf.zeros_like(pred_heatmaps_tmp)
        # pred_heatmaps_tmp_01 = tf.where(pred_heatmaps_tmp > 1, x=one_tmp, y=pred_heatmaps_tmp)
        # pred_heatmaps_tmp_01 = tf.where(pred_heatmaps_tmp_01 < 0, x=zero_tmp, y=pred_heatmaps_tmp_01)
        # loss_scoremap += tf.nn.l2_loss(pred_heatmaps_tmp_01 - scoremap)

        #用is loss 修正热度图
        pre_is_loss = tf.nn.softmax(pre_is_loss)

        pred_heatmaps_tmp_modi = tf.expand_dims(tf.expand_dims(pre_is_loss, axis=1), axis=1)*pred_heatmaps_tmp

        loss_scoremap_m += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_heatmaps_tmp_modi, labels=gt))

        pred_heatmaps_tmp = tf.nn.softmax(pred_heatmaps_tmp)
        pred_heatmaps_tmp_modi = tf.nn.softmax(pred_heatmaps_tmp_modi)

    loss_is_loss = loss_is_loss
    loss_scoremap = loss_scoremap
    loss_scoremap_m = loss_scoremap_m

    total_loss = loss_scoremap + loss_is_loss + loss_scoremap_m
    return total_loss, loss_is_loss, loss_scoremap, loss_scoremap_m, pred_heatmaps_tmp, pre_is_loss, pred_heatmaps_tmp_modi

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
        dataset_RHD = RHD(batchnum=params['batchsize'])

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(float(params['lr']), global_step,
                                                   decay_steps=10000, decay_rate=float(params['decay_rate']),
                                                   staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        tower_grads = []
        reuse_variable = False

        for i in range(params['gpus']):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("GPU_%d" % i):
                    #input_image, keypoint_xyz, keypoint_uv, input_heat, keypoint_vis, k, num_px_left_hand, num_px_right_hand \
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

                    input_image = input_image1
                    scoremap = scoremap1
                    is_loss = is_loss1


                    # 计算一个scoremap的loss
                    scoremap = tf.reduce_sum(scoremap, axis=-1)
                    one_scoremap = tf.ones_like(scoremap)
                    scoremap = tf.where(scoremap > 1, x=one_scoremap, y=scoremap)
                    scoremap = tf.expand_dims(scoremap, axis=-1)
                    scoremap = tf.concat([scoremap, 1-scoremap], axis=-1)

                    is_loss = tf.expand_dims(is_loss, axis=-1)
                    is_loss = tf.concat([is_loss, 1 - is_loss], axis=-1)

                    """
                    model, batchsize, input_image, scoremap, is_loss, reuse_variables=None
                    total_loss, loss_is_loss, loss_scoremap, pred_heatmaps_tmp, pre_is_loss, pred_heatmaps_tmp_01_modi
                    """
                    loss, loss_is_loss, loss_scoremap,loss_scoremap_m,\
                    preheat, pre_is_loss, pred_heatmaps_tmp_01_modi\
                        = get_loss_and_output(params['model'], params['batchsize'],
                                                input_image, scoremap, is_loss, reuse_variable)

                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        MOVING_AVERAGE_DECAY = 0.99
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(max_to_keep=10)

        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("loss", loss)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            checkpoint_path = os.path.join(params['modelpath'], training_name)
            model_name = '/model-4200'
            if checkpoint_path:
                saver.restore(sess, checkpoint_path+model_name)
                print("restore from " + checkpoint_path+model_name)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize']* 2 * params['gpus'])

            print("Start training...")
            for step in tqdm(range(total_step_num)):
                _, loss_value = sess.run([train_op, loss])
                if step % params['per_update_tensorboard_step'] == 0:
                    loss_v, loss_is_loss_v, loss_scoremap_v,loss_scoremap_m_v, \
                    input_image_v, scoremap_v, is_loss_v,\
                    preheat_v, pre_is_loss_v, pred_heatmaps_tmp_01_modi_v\
                        = sess.run([loss, loss_is_loss, loss_scoremap,loss_scoremap_m,
                         input_image, scoremap, is_loss,
                         preheat, pre_is_loss, pred_heatmaps_tmp_01_modi])

                    input_image_v = (input_image_v + 0.5) * 255
                    input_image_v = input_image_v.astype(np.int16)

                    fig = plt.figure(1)
                    plt.clf()
                    ax1 = fig.add_subplot(2, 4, 1)
                    ax1.imshow(input_image_v[0, :, :, :])#第一个batch的维度 hand1(0~31) back1(32~63)
                    ax1.axis('off')


                    ax2 = fig.add_subplot(2, 4, 2)
                    ax2.imshow(scoremap_v[0, :, :, 0])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                    ax2.axis('off')
                    ax2.set_title(str(is_loss_v[0, 0]))  # hand1 back1

                    ax6 = fig.add_subplot(2, 4, 6)
                    ax6.imshow(scoremap_v[0, :, :, 1])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                    if np.sum(scoremap_v[0, :, :, 1])==0:
                        print("zero")
                    ax6.axis('off')
                    ax6.set_title(str(is_loss_v[0, 1]))  # hand1 back1


                    ax3 = fig.add_subplot(2, 4, 3)
                    ax3.imshow(preheat_v[0, :, :, 0])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                    ax3.axis('off')
                    ax3.set_title(str(pre_is_loss_v[0, 0]))  # hand1 back1

                    ax7 = fig.add_subplot(2, 4, 7)
                    ax7.imshow(preheat_v[0, :, :, 1])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                    ax7.axis('off')
                    ax7.set_title(str(pre_is_loss_v[0, 1]))  # hand1 back1


                    ax4 = fig.add_subplot(2, 4, 4)
                    ax4.imshow(pred_heatmaps_tmp_01_modi_v[0, :, :, 0])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                    ax4.axis('off')
                    ax8 = fig.add_subplot(2, 4, 8)
                    ax8.imshow(pred_heatmaps_tmp_01_modi_v[0, :, :, 1])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                    ax8.axis('off')

                    ax5 = fig.add_subplot(2, 4, 5)
                    ax5.imshow(pred_heatmaps_tmp_01_modi_v[0, :, :, 0] - pred_heatmaps_tmp_01_modi_v[0, :, :, 1])  # 第一个batch的维度 hand1(0~31) back1(32~63)
                    ax5.axis('off')

                    plt.savefig(os.path.join(params['logpath']) + "/" + str(step).zfill(10) + "_.png")

                    print("loss:"+str(loss_v), " is_loss_loss:"+str(loss_is_loss_v)+" scoremap_loss:"
                          +str(loss_scoremap_v)+" scoremap_loss:"+str(loss_scoremap_m_v))


                    # save model
                if step % params['per_saved_model_step'] == 0:
                    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=step)

if __name__ == '__main__':
    tf.app.run()
