# coding-utf8
from __future__ import print_function, unicode_literals

import argparse

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image
import time
from mpl_toolkits.mplot3d import Axes3D
import cv2, os

from dataset_interface.RHD import RHD
from src import network_mv2_hourglass
from src.networks import get_network
from src.general import NetworkOps
ops = NetworkOps
def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor],
                                    name=name)
if __name__ == '__main__':

    model_name = 'model-29500'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    parser.add_argument('--model', type=str, default='mv2_hourglass', help='')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--checkpoint', type=str,
                        default='/home/chen/Documents/Mobile_hand/experiments/trained/mv2_hourglass_deep/models/mv2_hourglass_batch-32_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass/' + model_name,
                        help='checkpoint path')
    parser.add_argument('--output_node_names', type=str,
                        default='GPU_0/final_fxuz_Variable')  # ['GPU_0/final_r_Variable','GPU_0/final_x_Variable','GPU_0/final_y_Variable','GPU_0/final_z_Variable'])
    parser.add_argument('--output_graph', type=str,
                        default='/home/chen/Documents/Mobile_hand/experiments/trained/mv2_hourglass_deep/models/mv2_hourglass_batch-32_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass/' + model_name + '.pb',
                        help='output_freeze_path')

    args = parser.parse_args()
    i = 0
    batchsize = 1
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        dataset_RHD = RHD(batchnum=batchsize)

        with tf.device("/gpu:%d" % i):
            with tf.name_scope("GPU_%d" % i):
                #input_node = tf.placeholder(tf.float32, shape=[2, args.size, args.size, 3], name="input_image")

                # input_image, keypoint_xyz, keypoint_uv, input_heat, keypoint_vis, k, num_px_left_hand, num_px_right_hand \
                batch_data_all = dataset_RHD.get_batch_data
                input_image1 = batch_data_all[8]
                input_image2 = batch_data_all[10]
                hand_motion = batch_data_all[9]
                scoremap1 = batch_data_all[11]
                scoremap2 = batch_data_all[12]
                is_loss1 = batch_data_all[13]
                is_loss2 = batch_data_all[14]

                batch_data_all_back = dataset_RHD.coco_get_batch_back_data
                input_image1_back = batch_data_all_back[8]
                input_image2_back = batch_data_all_back[10]
                hand_motion_back = batch_data_all_back[9]
                scoremap1_back = batch_data_all_back[11]
                scoremap2_back = batch_data_all_back[12]
                is_loss1_back = batch_data_all_back[13]
                is_loss2_back = batch_data_all_back[14]

                input_image1 = tf.concat([input_image1, input_image1_back], 0)  # 第一个batch的维度 hand1 back1
                input_image2 = tf.concat([input_image2, input_image2_back], 0)
                input_image12 = tf.concat([input_image1, input_image2], 0)  # hand1 back1 hand2 back2
                input_image12.set_shape([batchsize * 4, 32, 32, 3])

                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    network_mv2_hourglass.N_KPOINTS = 1
                    _, pred_heatmaps_all12 = get_network('mv2_hourglass', input_image12, True)

                for batch_i in range(len(pred_heatmaps_all12)):
                    # 计算 isloss，用softmax计算 0~1}
                    is_loss_s = pred_heatmaps_all12[batch_i].get_shape().as_list()
                    pre_is_loss = tf.reshape(pred_heatmaps_all12[batch_i], [is_loss_s[0], -1])  # this is Bx16*16*1
                    out_chan_list = [32, 16, 8, 1]
                    for i, out_chan in enumerate(out_chan_list):
                        pre_is_loss = ops.fully_connected_relu(pre_is_loss, 'is_loss_fc_%d_%d' % (batch_i, i),
                                                               out_chan=out_chan, trainable=True)  # (128,1)
                    # 将pre_is_loss约束在01之间
                    one_pre_is_loss = tf.ones_like(pre_is_loss)
                    zero_pre_is_loss = tf.zeros_like(pre_is_loss)
                    pre_is_loss = tf.where(pre_is_loss > 1, x=one_pre_is_loss, y=pre_is_loss)
                    pre_is_loss = tf.where(pre_is_loss < 0, x=zero_pre_is_loss, y=pre_is_loss)

                    pred_heatmaps_tmp = upsample(pred_heatmaps_all12[batch_i], 2, name="upsample_for_hotmap_loss_%d" % batch_i)
                    one_tmp = tf.ones_like(pred_heatmaps_tmp)
                    pred_heatmaps_tmp = tf.where(pred_heatmaps_tmp > 1, x=one_tmp, y=pred_heatmaps_tmp)
                    # 用is loss 修正热度图
                    pred_heatmaps_tmp_ = tf.expand_dims(tf.expand_dims(pre_is_loss, axis=-1), axis=-1) * pred_heatmaps_tmp

                diffmap = []
                for batch_i in range(len(pred_heatmaps_all12)):  # hand1 back1 hand2 back2
                    diffmap.append(
                        pred_heatmaps_all12[batch_i][0:batchsize] - pred_heatmaps_all12[batch_i][
                                                                    batchsize:batchsize * 2])

                # diffmap_t 将4个阶段的输出，在通道数上整合
                for batch_i in range(len(diffmap)):
                    if batch_i == 0:
                        diffmap_t = diffmap[batch_i]
                    else:
                        diffmap_t = tf.concat([diffmap[batch_i], diffmap_t], axis=3)

                with tf.variable_scope("diff", reuse=False):
                    network_mv2_hourglass.N_KPOINTS = 1
                    _, pred_diffmap_all = get_network('mv2_hourglass', diffmap_t, True)
                losses = []
                for idx, pred_heat in enumerate(pred_diffmap_all):
                    # flatten
                    s = pred_heat.get_shape().as_list()
                    pred_heat = tf.reshape(pred_heat, [s[0], -1])  # this is Bx16*16*1
                    # x = tf.concat([x, hand_side], 1)

                    # pred_heat --> 3 params
                    out_chan_list = [32, 16, 8]
                    for i, out_chan in enumerate(out_chan_list):
                        pred_heat = ops.fully_connected_relu(pred_heat, 'fc_vp_%d_%d' % (idx, i), out_chan=out_chan,
                                                             trainable=True)

                    ux = ops.fully_connected(pred_heat, 'fc_vp_ux_%d' % idx, out_chan=1, trainable=True)
                    uy = ops.fully_connected(pred_heat, 'fc_vp_uy_%d' % idx, out_chan=1, trainable=True)
                    uz = ops.fully_connected(pred_heat, 'fc_vp_uz_%d' % idx, out_chan=1, trainable=True)
                    ur = ops.fully_connected(pred_heat, 'fc_vp_ur_%d' % idx, out_chan=1, trainable=True)
                    # output_node = tf.stack([ur[:, 0], ux[:, 0], uy[:, 0], uz[:, 0]], axis=1, name="final_rxyz")
                ufxuz = tf.concat(values=[ur, ux, uy, uz], axis=1, name='fxuz')

                # output_node_ur = tf.add(ur, 0, name='final_r_Variable')
                # output_node_ux = tf.add(ux, 0, name='final_x_Variable')
                # output_node_uy = tf.add(uy, 0, name='final_y_Variable')
                # output_node_uz = tf.add(uz, 0, name='final_z_Variable')
                output_node_ufxuz = tf.add(ufxuz, 0, name='final_fxuz_Variable')  # (1,4)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            saver = tf.train.Saver()
            saver.restore(sess, args.checkpoint)
            print("restore from " + args.checkpoint)

            cap = cv2.VideoCapture(0)
            # network input
            # Feed image list through network
            step = 0
            centerx = 160
            centery = 120

            path = "/media/chen/4CBEA7F1BEA7D1AE/VR715/"

            for i in range(500):
            #     image_raw1_crop = matplotlib.image.imread(path + str(step*2).zfill(5) + '.jpg')
            #     image_raw2_crop = matplotlib.image.imread(path + str(step*2+1).zfill(5) + '.jpg')
            #     image_raw1_crop = image_raw1_crop.astype('float') / 255.0 - 0.5
            #     image_raw1_crop = image_raw1_crop[centerx:centerx+32, centery:centery+32]
            #
            #     image_raw2_crop = image_raw2_crop.astype('float') / 255.0 - 0.5
            #     image_raw2_crop = image_raw2_crop[centerx:centerx + 32, centery:centery + 32]
            # # while(True):
            # #     # 获取相隔0.1s的两帧图片，做根据初始化或者网络输出结果，绘制矩形框，进行剪裁，reshape
            # #     _,image_raw1 = cap.read()
            # #     image_raw1 = scipy.misc.imresize(image_raw1, (240, 320))
            # #     time.sleep(0.1)
            # #     _,image_raw2 = cap.read()
            # #     image_raw2 = scipy.misc.imresize(image_raw2, (240, 320))
            # #         # 剪切 resize
            # #
            # #     size = 24
            # #     image_raw1_crop = np.array(image_raw1[centery - : centery + size, centerx - size: centerx + size])
            # #     image_raw2_crop = np.array(image_raw2[centery - size: centery + size, centerx - size: centerx + size])
            # #     image_raw1_crop = cv2.resize(image_raw1_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)
            # #     image_raw2_crop = cv2.resize(image_raw2_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)
            # #     first_point = (centerx-size, centery-size)
            # #     last_point = (centerx+size, centery+size)
            # #     cv2.rectangle(image_raw1, first_point, last_point, (0, 255, 0), 2)
            # #     cv2.rectangle(image_raw2, first_point, last_point, (0, 255, 0), 2)
            #
            #     image_raw12_crop = np.concatenate((image_raw1_crop[np.newaxis, :], image_raw2_crop[np.newaxis, :]), axis=0)

                input_image12_v ,preheat_v, pre_is_loss_v, output_node_ufxuz_ = sess.run([input_image12, pred_heatmaps_tmp, pre_is_loss, output_node_ufxuz])#, feed_dict={input_node: image_raw12_crop})
                output_node_ufxuz_ = output_node_ufxuz_[0]
                ur = round(output_node_ufxuz_[0], 3)
                ux = round(output_node_ufxuz_[1], 3)
                uy = round(output_node_ufxuz_[2], 3)
                uz = round(output_node_ufxuz_[3], 3)

                # pt1 = (centerx, centery)
                # pt2 = (centerx-int(ux*48), centery-int(uy*48))
                # cv2.arrowedLine(image_raw1, pt1, pt2, (255, 0, 0), 2)
                #
                # if (abs(ux) > 0.12):
                #     centerx=centerx - int(ux*48)
                #     if(centerx<=24 or centerx>=320-24):
                #         centerx = 160
                # if (abs(uy) > 0.12):
                #     centery=centery - int(uy*48)
                #     if(centery<=24 or centery>=240-24):
                #        centery = 120
                # visualize
                fig = plt.figure(1)
                fig.clear()
                ax1 = fig.add_subplot(241)
                ax2 = fig.add_subplot(242)
                ax3 = fig.add_subplot(243)
                ax4 = fig.add_subplot(244)
                ax5 = fig.add_subplot(245)
                ax6 = fig.add_subplot(246)
                ax7 = fig.add_subplot(247)
                ax8 = fig.add_subplot(248)


                image_raw1_crop = (input_image12_v[0,:,:,:] + 0.5) * 255
                image_raw1_crop = image_raw1_crop.astype(np.int16)
                image_raw2_crop = (input_image12_v[1,:,:,:] + 0.5) * 255
                image_raw2_crop = image_raw2_crop.astype(np.int16)
                image_raw3_crop = (input_image12_v[2,:,:,:] + 0.5) * 255
                image_raw3_crop = image_raw3_crop.astype(np.int16)
                image_raw4_crop = (input_image12_v[3,:,:,:] + 0.5) * 255
                image_raw4_crop = image_raw4_crop.astype(np.int16)

                ax1.imshow(image_raw1_crop)
                ax2.imshow(image_raw2_crop)
                ax3.imshow(image_raw3_crop)
                ax4.imshow(image_raw4_crop)
                ax5.imshow(preheat_v[0, :, :, 0])
                ax6.imshow(preheat_v[1, :, :, 0])
                ax7.imshow(preheat_v[2, :, :, 0])
                ax8.imshow(preheat_v[3, :, :, 0])
                ax5.set_title(str(pre_is_loss_v[0]))
                ax6.set_title(str(pre_is_loss_v[1]))
                ax7.set_title(str(pre_is_loss_v[2]))
                ax8.set_title(str(pre_is_loss_v[3]))

                # ax3.imshow(preheat_v[0, :, :, 0])
                # ax3.set_title(str(pre_is_loss_[0]))
                # ax4.imshow(preheat_v[1, :, :, 0])
                # ax4.set_title(str(pre_is_loss_[1]))
                #
                # ax5.plot([0, ux], [0, uy], label="predict", color='blue')
                # ax5.set_xlim((-1, 1))
                # ax5.set_ylim((1, -1))
                # ax5.grid(True)

                plt.savefig("/home/chen/Documents/Mobile_hand/experiments/varify/image/valid_on_cam/"+
                            str(step).zfill(5)+".jpg")
                plt.pause(0.01)
                step = step+1
