# coding-utf8
from __future__ import print_function, unicode_literals

import argparse

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import cv2, os
from src import network_mv2_hourglass
from src.networks import get_network
from src.general import NetworkOps
ops = NetworkOps
def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor],
                                    name=name)
if __name__ == '__main__':

    model_name = 'model-173500'
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
        with tf.device("/gpu:%d" % i):
            with tf.name_scope("GPU_%d" % i):
                input_node = tf.placeholder(tf.float32, shape=[2, args.size, args.size, 3], name="input_image")
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    network_mv2_hourglass.N_KPOINTS = 1
                    _, pred_heatmaps_all12 = get_network('mv2_hourglass', input_node, True)
                diffmap = []
                for batch_i in range(len(pred_heatmaps_all12)):
                    diffmap.append(
                        pred_heatmaps_all12[batch_i][0:batchsize] - pred_heatmaps_all12[batch_i][
                                                                    batchsize:batchsize * 2])
                preheat = upsample(pred_heatmaps_all12[-1], 2, name="upsample_for_hotmap_loss_%d" % batch_i)
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
            while(True):
                # 获取相隔0.1s的两帧图片，做根据初始化或者网络输出结果，绘制矩形框，进行剪裁，reshape
                _,image_raw1 = cap.read()
                image_raw1 = scipy.misc.imresize(image_raw1, (240, 320))
                time.sleep(0.1)
                _,image_raw2 = cap.read()
                image_raw2 = scipy.misc.imresize(image_raw2, (240, 320))
                    # 剪切 resize

                size = 24
                image_raw1_crop = np.array(image_raw1[centery - size: centery + size, centerx - size: centerx + size])
                image_raw2_crop = np.array(image_raw2[centery - size: centery + size, centerx - size: centerx + size])
                image_raw1_crop = cv2.resize(image_raw1_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)
                image_raw2_crop = cv2.resize(image_raw2_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)
                first_point = (centerx-size, centery-size)
                last_point = (centerx+size, centery+size)
                cv2.rectangle(image_raw1, first_point, last_point, (0, 255, 0), 2)
                cv2.rectangle(image_raw2, first_point, last_point, (0, 255, 0), 2)

                image_raw12_crop = np.concatenate((image_raw1_crop[np.newaxis, :], image_raw2_crop[np.newaxis, :]), axis=0)
                image_raw12_crop = image_raw12_crop.astype('float') / 255.0 - 0.5

                preheat_v, output_node_ufxuz_ = sess.run([preheat, output_node_ufxuz], feed_dict={input_node: image_raw12_crop})
                output_node_ufxuz_ = output_node_ufxuz_[0]
                ur = round(output_node_ufxuz_[0], 3)
                ux = round(output_node_ufxuz_[1], 3)
                uy = round(output_node_ufxuz_[2], 3)
                uz = round(output_node_ufxuz_[3], 3)


                pt1 = (centerx, centery)
                pt2 = (centerx-int(ux*48), centery-int(uy*48))
                cv2.arrowedLine(image_raw1, pt1, pt2, (255, 0, 0), 2)

                if (abs(ux) > 0.12):
                    centerx=centerx - int(ux*48)
                    if(centerx<=24 or centerx>=320-24):
                        centerx = 160
                if (abs(uy) > 0.12):
                    centery=centery - int(uy*48)
                    if(centery<=24 or centery>=240-24):
                       centery = 120
                # visualize
                fig = plt.figure(1)
                fig.clear()
                ax1 = fig.add_subplot(321)
                ax2 = fig.add_subplot(322)
                ax3 = fig.add_subplot(323)
                ax4 = fig.add_subplot(324)
                ax5 = fig.add_subplot(325)
                ax5.imshow(preheat_v[0, :, :, 0])
                ax6 = fig.add_subplot(326)
                ax6.imshow(preheat_v[1, :, :, 0])

                ax1.imshow(image_raw1)
                ax2.imshow(image_raw2)
                ax3.imshow(image_raw1_crop)
                ax4.imshow(image_raw2_crop)
                ax1.set_title('ur' + str(ur))
                ax2.set_title('ux' + str(ux))
                ax3.set_title('uy' + str(uy))
                ax4.set_title('uz' + str(uz))
                plt.savefig("/home/chen/Documents/Mobile_hand/experiments/trained/mv2_hourglass_deep/log/valid_on_cam/"+
                            str(step).zfill(5)+".jpg")
                plt.pause(0.01)
                step = step+1
