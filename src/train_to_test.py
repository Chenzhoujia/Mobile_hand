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
def get_loss_and_output(model, batchsize, input_image1, input_image2, reuse_variables=None):
    losses = []

    # 叠加在batch上重用特征提取网络
    input_image12 = tf.concat([input_image1, input_image2], 0)  #hand1 back1 hand2 back2
    #input_image12.set_shape([batchsize * 4, 32, 32, 3])
    #input_image12 = tf.add(input_image12, 0, name='input_image')
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        network_mv2_hourglass.N_KPOINTS = 1
        _, pred_heatmaps_all12 = get_network(model, input_image12, True) #第一个batch的维度 hand1 back1 hand2 back2

    for loss_i in range(len(pred_heatmaps_all12)):
        # 计算 isloss，用softmax计算 0~1}
        # is_loss_s = pred_heatmaps_all12[loss_i].get_shape().as_list()
        pre_is_loss = tf.reshape(pred_heatmaps_all12[loss_i], [batchsize*4, -1])  # this is Bx16*16*1
        out_chan_list = [32, 16, 8, 1]
        for i, out_chan in enumerate(out_chan_list):
            pre_is_loss = ops.fully_connected_relu(pre_is_loss, 'is_loss_fc_%d_%d' % (loss_i, i), out_chan=out_chan, trainable=True)#(128,1)
        #将pre_is_loss约束在01之间
        one_pre_is_loss = tf.ones_like(pre_is_loss)
        zero_pre_is_loss = tf.zeros_like(pre_is_loss)
        pre_is_loss = tf.where(pre_is_loss > 1, x=one_pre_is_loss, y=pre_is_loss)
        pre_is_loss = tf.where(pre_is_loss < 0, x=zero_pre_is_loss, y=pre_is_loss)

        #pre_is_loss = tf.nn.softmax(pre_is_loss)

        # 计算热度图
        scale = 2
        pred_heatmaps_tmp = upsample(pred_heatmaps_all12[loss_i], scale, name="upsample_for_hotmap_loss_%d" % loss_i)
        one_tmp = tf.ones_like(pred_heatmaps_tmp)
        pred_heatmaps_tmp = tf.where(pred_heatmaps_tmp > 1, x=one_tmp, y=pred_heatmaps_tmp)

        #用is loss 修正热度图
        pred_heatmaps_tmp_ = tf.expand_dims(tf.expand_dims(pre_is_loss, axis=-1), axis=-1)*pred_heatmaps_tmp

    diffmap = []
    for batch_i in range(len(pred_heatmaps_all12)):   #hand1 back1 hand2 back2
        diffmap.append(pred_heatmaps_all12[batch_i][0:batchsize*2]-pred_heatmaps_all12[batch_i][batchsize*2:batchsize*4])

    #diffmap_t 将4个阶段的输出，在通道数上整合
    for batch_i in range(len(diffmap)):
        if batch_i==0:
            diffmap_t = diffmap[batch_i]
        else:
            diffmap_t = tf.concat([diffmap[batch_i], diffmap_t], axis=3) #hand12 back12

    with tf.variable_scope("diff", reuse=reuse_variables):
        network_mv2_hourglass.N_KPOINTS = 1
        _, pred_diffmap_all = get_network(model, diffmap_t, True)
    for idx, pred_heat in enumerate(pred_diffmap_all):
        # flatten
        s = pred_heat.get_shape().as_list()
        pred_heat = tf.reshape(pred_heat, [s[0], -1])  # this is Bx16*16*1
        #x = tf.concat([x, hand_side], 1)

        # pred_heat --> 3 params
        out_chan_list = [32, 16, 8]
        for i, out_chan in enumerate(out_chan_list):
            pred_heat = ops.fully_connected_relu(pred_heat, 'fc_vp_%d_%d' %(idx,i), out_chan=out_chan, trainable=True)
            evaluation = tf.placeholder_with_default(True, shape=())
            pred_heat = pred_heat# ops.dropout(pred_heat, 0.95, evaluation)

        ux = ops.fully_connected(pred_heat, 'fc_vp_ux_%d' % idx, out_chan=1, trainable=True)
        uy = ops.fully_connected(pred_heat, 'fc_vp_uy_%d' % idx, out_chan=1, trainable=True)
        uz = ops.fully_connected(pred_heat, 'fc_vp_uz_%d' % idx, out_chan=1, trainable=True)
        ur = ops.fully_connected(pred_heat, 'fc_vp_ur_%d' % idx, out_chan=1, trainable=True)

    ufxuz = tf.concat(values=[ur, ux, uy, uz], axis=1, name='fxuz')

    return ur, ux, uy, uz, ufxuz, pred_heatmaps_tmp,pred_heatmaps_tmp_, pre_is_loss

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
        '32',
        params['lr'],
        gpus,
        params['gpus'],
        params['input_width'], params['input_height'],
        "..-experiments-mv2_hourglass"
    )

    with tf.Graph().as_default(), tf.device("/cpu:0"):
        dataset_RHD = RHD(batchnum=params['batchsize'])

        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(float(params['lr']), global_step,
        #                                            decay_steps=10000, decay_rate=float(params['decay_rate']),
        #                                            staircase=True)
        # opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        # tower_grads = []
        reuse_variable = False

        for i in range(params['gpus']):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("GPU_%d" % i):
                    #input_image, keypoint_xyz, keypoint_uv, input_heat, keypoint_vis, k, num_px_left_hand, num_px_right_hand \
                    batch_data_all = dataset_RHD.get_batch_data
                    input_image1 = batch_data_all[8]
                    input_image2 = batch_data_all[10]
                    # hand_motion = batch_data_all[9]
                    # scoremap1 = batch_data_all[11]
                    # scoremap2 = batch_data_all[12]
                    # is_loss1 = batch_data_all[13]
                    # is_loss2 = batch_data_all[14]


                    batch_data_all_back = dataset_RHD.coco_get_batch_back_data
                    input_image1_back = batch_data_all_back[8]
                    input_image2_back = batch_data_all_back[10]
                    # hand_motion_back = batch_data_all_back[9]
                    # scoremap1_back = batch_data_all_back[11]
                    # scoremap2_back = batch_data_all_back[12]
                    # is_loss1_back = batch_data_all_back[13]
                    # is_loss2_back = batch_data_all_back[14]

                    input_image1 = tf.concat([input_image1, input_image1_back], 0) #第一个batch的维度 hand1 back1
                    input_image2 = tf.concat([input_image2, input_image2_back], 0)
                    # hand_motion = tf.concat([hand_motion, hand_motion_back], 0)#第一个batch的维度 hand12 back12
                    # scoremap1 = tf.concat([scoremap1, scoremap1_back], 0)
                    # scoremap2 = tf.concat([scoremap2, scoremap2_back], 0)
                    # is_loss1 = tf.concat([is_loss1, is_loss1_back], 0)
                    # is_loss2 = tf.concat([is_loss2, is_loss2_back], 0)
                    """
                    total_loss, motion_loss*0.00001, loss_scoremap*0.001, loss_is_loss,\
                               ur, ux, uy, uz, ufxuz, pred_heatmaps_tmp, pre_is_loss, is_loss12
                    """
                    ur, ux, uy, uz, ufxuz, preheat, preheat_, pre_is_loss\
                        = get_loss_and_output(params['model'], params['batchsize'], input_image1, input_image2, reuse_variable)

        saver = tf.train.Saver(max_to_keep=10)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            checkpoint_path = os.path.join(params['modelpath'], training_name)
            model_name = '/model-29500'
            if checkpoint_path:
                saver.restore(sess, checkpoint_path+model_name)
                print("restore from " + checkpoint_path+model_name)

            summary_writer = tf.summary.FileWriter(os.path.join(params['logpath'], training_name), sess.graph)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize'] * params['gpus'])
            print("Start training...")
            for step in tqdm(range(total_step_num)):
                valid_input_image1, valid_input_image2, \
                ur_v, ux_v, uy_v, uz_v, preheat_v, preheat_m_v, pre_is_loss_v = sess.run(
                    [input_image1, input_image2, ur, ux, uy, uz, preheat, preheat_, pre_is_loss])

                valid_input_image1 = (valid_input_image1 + 0.5) * 255
                valid_input_image1 = valid_input_image1.astype(np.int16)

                valid_input_image2 = (valid_input_image2 + 0.5) * 255
                valid_input_image2 = valid_input_image2.astype(np.int16)

                fig = plt.figure(1)
                plt.clf()
                ax1 = fig.add_subplot(3,4,1)
                ax1.imshow(valid_input_image1[0, :, :, :])#第一个batch的维度 hand1(0~31) back1(32~63)
                ax1.axis('off')
                ax2 = fig.add_subplot(3,4,2)
                ax2.imshow(valid_input_image2[0, :, :, :])#第一个batch的维度 hand2 back2
                ax2.axis('off')
                ax3 = fig.add_subplot(3,4,3)
                ax3.imshow(valid_input_image1[1*params['batchsize'], :, :, :])#第一个batch的维度 hand1 back1
                ax3.axis('off')
                ax4 = fig.add_subplot(3,4,4)
                ax4.imshow(valid_input_image2[1*params['batchsize'], :, :, :])#第一个batch的维度 hand2 back2
                ax4.axis('off')

                ax5 = fig.add_subplot(3, 4, 5)
                ax5.imshow(np.sum(preheat_v[0], axis=-1))  # 第一个batch的维度 hand1(0~31) back1(32~63)
                ax5.axis('off')
                ax5.set_title(str(pre_is_loss_v[0]))  # hand1 back1 hand2 back2
                ax6 = fig.add_subplot(3, 4, 6)
                ax6.imshow(np.sum(preheat_v[2*params['batchsize']], axis=-1))  # 第一个batch的维度 hand2 back2
                ax6.axis('off')
                ax6.set_title(str(pre_is_loss_v[2*params['batchsize']]))
                ax7 = fig.add_subplot(3, 4, 7)
                ax7.imshow(np.sum(preheat_v[1*params['batchsize']], axis=-1))  # 第一个batch的维度 hand1 back1
                ax7.axis('off')
                ax7.set_title(str(pre_is_loss_v[1*params['batchsize']]))
                ax8 = fig.add_subplot(3, 4, 8)
                ax8.imshow(np.sum(preheat_v[3*params['batchsize']], axis=-1))  # 第一个batch的维度 hand2 back2
                ax8.axis('off')
                ax8.set_title(str(pre_is_loss_v[3*params['batchsize']]))

                ax9 = fig.add_subplot(3,4,9)
                ax9.imshow(np.sum(preheat_m_v[0*params['batchsize']], axis=-1))#hand1 back1 hand2 back2
                ax9.axis('off')
                ax10 = fig.add_subplot(3,4,10)
                ax10.imshow(np.sum(preheat_m_v[2*params['batchsize']], axis=-1))
                ax10.axis('off')
                ax11 = fig.add_subplot(3,4,11)
                ax11.imshow(np.sum(preheat_m_v[1*params['batchsize']], axis=-1))
                ax11.axis('off')
                ax12 = fig.add_subplot(3,4,12)
                ax12.imshow(np.sum(preheat_m_v[3*params['batchsize']], axis=-1))
                ax12.axis('off')
                plt.savefig(os.path.join(params['logpath'], training_name) + "/" + str(step).zfill(10) + ".png")


                fig2 = plt.figure(2)
                plt.clf()
                ax13 = fig2.add_subplot(2,4,2)#hand12 back12
                ax13.plot([0, ux_v[0]], [0, uy_v[0]], label="predict", color='blue')
                ax13.set_xlim((-1, 1))
                ax13.set_ylim((1, -1))
                ax13.grid(True)

                ax15 = fig2.add_subplot(2,4,3)
                ax15.plot([0, ux_v[1*params['batchsize']]], [0, uy_v[1*params['batchsize']]], label="predict", color='blue')
                ax15.set_xlim((-1, 1))
                ax15.set_ylim((-1, 1))
                ax15.grid(True)
                ax16 = fig2.add_subplot(2,4,5)
                ax16.imshow(np.sum(preheat_m_v[0], axis=-1))#hand1 back1 hand2 back2
                ax16.axis('off')
                ax17 = fig2.add_subplot(2,4,6)
                ax17.imshow(np.sum(preheat_m_v[2*params['batchsize']], axis=-1))
                ax17.axis('off')
                ax18 = fig2.add_subplot(2,4,7)
                ax18.imshow(np.sum(preheat_m_v[1*params['batchsize']], axis=-1))
                ax18.axis('off')
                ax19 = fig2.add_subplot(2,4,8)
                ax19.imshow(np.sum(preheat_m_v[3*params['batchsize']], axis=-1))
                ax19.axis('off')

                plt.savefig(os.path.join(params['logpath'], training_name) + "/" + str(step).zfill(10) + "_.png")

if __name__ == '__main__':
    tf.app.run()
