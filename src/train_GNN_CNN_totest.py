# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import sys
import time
import numpy as np
from numpy import unravel_index

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from tqdm import tqdm
from utils.general import LearningRateScheduler, load_weights_from_snapshot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from RGB_db_interface.GANerate import GANerate, plot_hand
from dataset_interface.RHD import RHD
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def main(argv=None):
    train_para = {'lr': [1e-4, 1e-5, 1e-6],
                  'lr_iter': [10000, 20000],
                  'max_iter': 30000,
                  'show_loss_freq': 1000,
                  'snapshot_freq': 5000,
                  'snapshot_dir': 'snapshots_posenet'}

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.train.start_queue_runners(sess=sess)

    # get dataset
    dataset_GANerate = GANerate(batchnum=32)
    image_crop_eval, keypoint_uv21_eval, keypoint_uv_heatmap_eval, keypoint_xyz21_normed_eval = dataset_GANerate.get_batch_data_eval

    # build network
    evaluation = tf.placeholder_with_default(True, shape=())
    net = ColorHandPose3DNetwork()

    image_crop_eval = tf.add(image_crop_eval, 0,
                                     name='input_node_representations')
    keypoints_scoremap_eval = net.inference_pose2d(image_crop_eval, train=True)
    s = keypoint_uv_heatmap_eval.get_shape().as_list()
    keypoints_scoremap_eval = [tf.image.resize_images(x, (s[1], s[2])) for x in keypoints_scoremap_eval]

    # Loss
    loss_eval = 0.0
    for i, pred_item in enumerate(keypoints_scoremap_eval):
        loss_eval += tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(pred_item - keypoint_uv_heatmap_eval), [1, 2])))
    keypoints_scoremap_eval = keypoints_scoremap_eval[-1]
    keypoints_scoremap_eval = tf.add(keypoints_scoremap_eval, 0,
                                     name='final_output_node_representations')
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    # occupy gpu gracefully
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init.run()
        checkpoint_path = './snapshots_posenet'
        model_name = 'model-42'
        if checkpoint_path:
            saver = tf.train.Saver(max_to_keep=10)
            saver.restore(sess, checkpoint_path+'/'+model_name)
            print("restore from " + checkpoint_path+'/'+model_name)
        create_pb = True
        if create_pb:
            input_graph_def = sess.graph.as_graph_def()
            variable_names = [v.name for v in input_graph_def.node]
            print('==================Model Analysis Report variable_names======================')
            print(variable_names)
            print('==================Model Analysis Report operations======================')
            for op in sess.graph.get_operations():
                print(str(op.name))
            stats_graph(sess.graph)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session
                input_graph_def,  # input_graph_def is useful for retrieving the nodes
                'final_output_node_representations'.split(",")
            )

            with tf.gfile.FastGFile(checkpoint_path+'/'+model_name+ ".pb", "wb") as f:
                f.write(output_graph_def.SerializeToString())

        print("Start testing...")
        path = './snapshots_posenet/baseline'
        import matplotlib.image
        loss_eval_v = 0.0
        loss_piex_save = 0.0
        for one_epoch in tqdm(range(100)):
            image, heatmap, heatmap_pre, keypoint_uv21, loss_eval_v = sess.run([image_crop_eval, keypoint_uv_heatmap_eval, keypoints_scoremap_eval,
                                                                  keypoint_uv21_eval, loss_eval])
            image = (image + 0.5) * 255
            image = image.astype(np.int16)

            #根据热度图计算最大的下标
            keypoint_uv21_pre = np.zeros_like(keypoint_uv21)
            for i in range(heatmap_pre.shape[0]):
                for j in range(heatmap_pre.shape[-1]):
                    heatmap_pre_tmp = heatmap_pre[i,:,:,j]
                    cor_tmp = unravel_index(heatmap_pre_tmp.argmax(), heatmap_pre_tmp.shape)
                    keypoint_uv21_pre[i,j,0] = cor_tmp[1]
                    keypoint_uv21_pre[i,j,1] = cor_tmp[0]

            loss_piex = keypoint_uv21_pre - keypoint_uv21
            loss_piex = np.sqrt(np.square(loss_piex[:,:,0]) + np.square(loss_piex[:,:,1]))
            loss_piex_save = loss_piex_save + np.mean(loss_piex)

            # visualize
            fig = plt.figure(1)
            plt.clf()
            ax1 = fig.add_subplot(221)
            ax1.imshow(image[0])
            plot_hand(keypoint_uv21[0], ax1)

            ax3 = fig.add_subplot(223)
            ax3.imshow(image[0])
            ax3.set_title(str(loss_piex[0,:].astype(np.int32)), fontsize=5)
            plot_hand(keypoint_uv21_pre[0], ax3)
            plot_hand(keypoint_uv21[0], ax3)

            ax2 = fig.add_subplot(222)
            ax4 = fig.add_subplot(224)
            ax2.imshow(np.sum(heatmap[0], axis=-1))  # 第一个batch的维度 hand1(0~31) back1(32~63)
            ax2.scatter(keypoint_uv21[0, :, 0], keypoint_uv21[0, :, 1], s=10, c='k', marker='.')
            ax4.imshow(np.sum(heatmap_pre[0], axis=-1))  # 第一个batch的维度 hand1(0~31) back1(32~63)
            ax4.scatter(keypoint_uv21_pre[0, :, 0], keypoint_uv21_pre[0, :, 1], s=10, c='k', marker='.')

            plt.savefig(path+'/image/' + str(one_epoch).zfill(5) + '.png')
        loss_eval_v = loss_eval_v / 100
        loss_piex_save = loss_piex_save/100
        print(loss_piex_save) #4.472415127649567

if __name__ == '__main__':
    tf.app.run()
