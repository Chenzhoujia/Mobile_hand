# coding=UTF-8
import pickle

import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dataset_interface.RHD_canonical_trafo import canonical_trafo, flip_right_hand
from dataset_interface.RHD_general import crop_image_from_xy
from dataset_interface.RHD_relative_trafo import bone_rel_trafo
from dataset_interface.base_interface import BaseDataset
import tensorflow as tf
import time

import os.path
import json
import matplotlib


def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure.
    W,
    T0, T1, T2, T3,
    I0, I1, I2, I3,
     M0, M1, M2, M3,
     R0, R1, R2, R3,
     L0, L1, L2, L3.
    """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 1), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_hw[connection[0], :]
        coord2 = coords_hw[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], color_fixed, linewidth=linewidth)


def plot_hand_3d(coords_xyz, axis, color_fixed=None, linewidth='1'):
    """ Plots a hand stick figure into a matplotlib figure. """
    colors = np.array([[0., 0., 0.5],
                       [0., 0., 0.73172906],
                       [0., 0., 0.96345811],
                       [0., 0.12745098, 1.],
                       [0., 0.33137255, 1.],
                       [0., 0.55098039, 1.],
                       [0., 0.75490196, 1.],
                       [0.06008855, 0.9745098, 0.90765338],
                       [0.22454143, 1., 0.74320051],
                       [0.40164453, 1., 0.56609741],
                       [0.56609741, 1., 0.40164453],
                       [0.74320051, 1., 0.22454143],
                       [0.90765338, 1., 0.06008855],
                       [1., 0.82861293, 0.],
                       [1., 0.63979666, 0.],
                       [1., 0.43645606, 0.],
                       [1., 0.2476398, 0.],
                       [0.96345811, 0.0442992, 0.],
                       [0.73172906, 0., 0.],
                       [0.5, 0., 0.]])

    # define connections and colors of the bones
    bones = [((0, 1), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 5), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 9), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 13), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 17), colors[16, :]),
             ((20, 19), colors[17, :]),
             ((19, 18), colors[18, :]),
             ((18, 17), colors[19, :])]

    for connection, color in bones:
        coord1 = coords_xyz[connection[0], :]
        coord2 = coords_xyz[connection[1], :]
        coords = np.stack([coord1, coord2])
        if color_fixed is None:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, linewidth=linewidth)
        else:
            axis.plot(coords[:, 0], coords[:, 1], coords[:, 2], color_fixed, linewidth=linewidth)

    axis.view_init(azim=-90., elev=90.)

def create_multiple_gaussian_map(coords_uv, output_size, sigma = 25.0, valid_vec=None):
    """ Creates a map of size (output_shape[0], output_shape[1]) at (center[0], center[1])
        with variance sigma for multiple coordinates."""
    with tf.name_scope('create_multiple_gaussian_map'):
        sigma = tf.cast(sigma, tf.float32)
        assert len(output_size) == 2
        s = coords_uv.get_shape().as_list()
        coords_uv = tf.cast(coords_uv, tf.int32)
        if valid_vec is not None:
            valid_vec = tf.cast(valid_vec, tf.float32)
            valid_vec = tf.squeeze(valid_vec)
            cond_val = tf.greater(valid_vec, 0.5)
        else:
            cond_val = tf.ones_like(coords_uv[:, 0], dtype=tf.float32)
            cond_val = tf.greater(cond_val, 0.5)

        cond_1_in = tf.logical_and(tf.less(coords_uv[:, 1], output_size[0]-1), tf.greater(coords_uv[:, 1], 0))
        cond_2_in = tf.logical_and(tf.less(coords_uv[:, 0], output_size[1]-1), tf.greater(coords_uv[:, 0], 0))
        cond_in = tf.logical_and(cond_1_in, cond_2_in)
        cond = tf.logical_and(cond_val, cond_in)

        coords_uv = tf.cast(coords_uv, tf.float32)

        # create meshgrid
        x_range = tf.expand_dims(tf.range(output_size[0]), 1)
        y_range = tf.expand_dims(tf.range(output_size[1]), 0)

        X = tf.cast(tf.tile(x_range, [1, output_size[1]]), tf.float32)
        Y = tf.cast(tf.tile(y_range, [output_size[0], 1]), tf.float32)

        X.set_shape((output_size[0], output_size[1]))
        Y.set_shape((output_size[0], output_size[1]))

        X = tf.expand_dims(X, -1)
        Y = tf.expand_dims(Y, -1)

        X_b = tf.tile(X, [1, 1, s[0]])
        Y_b = tf.tile(Y, [1, 1, s[0]])

        X_b -= coords_uv[:, 1]
        Y_b -= coords_uv[:, 0]

        dist = tf.square(X_b) + tf.square(Y_b)

        scoremap = tf.exp(-dist / tf.square(sigma)) * tf.cast(cond, tf.float32)

        return scoremap

class CMU(BaseDataset):
    def __init__(self, path="/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/CMUDexter/data",
                 batchnum=4):
        # 将标签加载到内存中
        """
        For every frame, the following data is provided:
        - color: 			24-bit color image of the hand (cropped)
        - joint_pos: 		3D joint positions relative to the middle MCP joint. The values are normalized such that the length between middle finger MCP and wrist is 1.
                            The positions are organized as a linear concatenation of the x,y,z-position of every joint (joint1_x, joint1_y, joint1_z, joint2_x, ...).
                            The order of the 21 joints is as follows: W, T0, T1, T2, T3, I0, I1, I2, I3, M0, M1, M2, M3, R0, R1, R2, R3, L0, L1, L2, L3.
                            Please also see joints.png for a visual explanation.
        - joint2D:			2D joint positions in u,v image coordinates. The positions are organized as a linear concatenation of the u,v-position of every joint (joint1_u, joint1_v, joint2_u, …).
        - joint_pos_global:	3D joint positions in the coordinate system of the original camera (before cropping)
        - crop_params:		cropping parameters that were used to generate the color image (256 x 256 pixels) from the original image (640 x 480 pixels),
                            specified as top left corner of the bounding box (u,v) and a scaling factor
        """
        super().__init__(path)

        # Input data paths
        folderPath = '/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/CMU3_hand143_panopticdb/'  # Put your local path here
        jsonPath = folderPath + 'hands_v143_14817.json'

        with open(jsonPath, 'r') as fid:
            dat_all = json.load(fid)
            dat_all = dat_all['root']
        self.GAN_color_path = []
        self.joint2D = np.zeros([len(dat_all),21,2])
        for dat_i in tqdm(range(len(dat_all))):
            self.GAN_color_path.append(folderPath + dat_all[dat_i]['img_paths'])
            self.joint2D[dat_i] = np.array(dat_all[dat_i]['joint_self'])[:,:2]

        self.sample_num = len(self.GAN_color_path)
        self.imagefilenames = tf.constant(self.GAN_color_path)

        # 创建训练数据集

        dataset = tf.data.Dataset.from_tensor_slices((self.imagefilenames, self.joint2D))
        dataset = dataset.map(CMU._parse_function)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=32)
        self.dataset = dataset.batch(batchnum)
        #self.iterator = self.dataset.make_initializable_iterator() sess.run(dataset_RHD.iterator.initializer)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.get_batch_data = self.iterator.get_next()
        #
        # # 创建测试数据集
        # dataset_eval = tf.data.Dataset.from_tensor_slices((self.imagefilenames[train_num:], self.joint2D[train_num:], self.joint_pos[train_num:]))
        # dataset_eval = dataset_eval.map(CMU._parse_function)
        # dataset_eval = dataset_eval.repeat()
        # dataset_eval = dataset_eval.shuffle(buffer_size=320)
        # self.dataset_eval = dataset_eval.batch(batchnum)
        # #self.iterator = self.dataset.make_initializable_iterator() sess.run(dataset_RHD.iterator.initializer)
        # self.iterator_eval = self.dataset_eval.make_one_shot_iterator()
        # self.get_batch_data_eval = self.iterator_eval.get_next()


    @staticmethod
    def visualize_data(image_crop, keypoint_uv21, keypoint_uv_heatmap):
        # get info from annotation dictionary

        image_crop = (image_crop + 0.5) * 255
        image_crop = image_crop.astype(np.int16)


        # visualize
        fig = plt.figure(1)
        plt.clf()
        ax1 = fig.add_subplot(221)

        ax1.imshow(image_crop)
        #plot_hand(keypoint_uv21, ax1)
        ax1.scatter(keypoint_uv21[:, 0], keypoint_uv21[:, 1], s=10, c='k', marker='.')
        #ax1.scart(keypoint_uv21[:, 0], keypoint_uv21[:, 1], color=color, linewidth=1)

        ax3 = fig.add_subplot(223)
        ax3.imshow(np.sum(keypoint_uv_heatmap, axis=-1))  # 第一个batch的维度 hand1(0~31) back1(32~63)
        ax3.scatter(keypoint_uv21[:, 0], keypoint_uv21[:, 1], s=10, c='k', marker='.')
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        plt.savefig('/tmp/image/'+now+'.png')

    @staticmethod
    def _parse_function(imagefilename, keypoint_uv):
        # 数据的基本处理
        image_size = (1080, 1920)
        image_string = tf.read_file(imagefilename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_decoded = tf.image.resize_images(image_decoded, [1080, 1920], method=0)
        image_decoded.set_shape([image_size[0],image_size[1],3])
        image = tf.cast(image_decoded, tf.float32)
        image = image / 255.0 - 0.5

        # 图片剪切
        crop_center = keypoint_uv21[12, ::-1]

        # catch problem, when no valid kp available (happens almost never)
        crop_center = tf.cond(tf.reduce_all(tf.is_finite(crop_center)), lambda: crop_center,
                              lambda: tf.constant([0.0, 0.0]))
        crop_center.set_shape([2, ])

        if crop_center_noise:
            noise = tf.truncated_normal([2], mean=0.0, stddev=crop_center_noise_sigma)
            crop_center += noise

        crop_scale_noise = tf.constant(1.0)
        if crop_scale_noise_:
            crop_scale_noise = tf.squeeze(tf.random_uniform([1], minval=1.0, maxval=1.2))

        # select visible coords only
        kp_coord_h = tf.boolean_mask(keypoint_uv21[:, 1], keypoint_vis21)
        kp_coord_w = tf.boolean_mask(keypoint_uv21[:, 0], keypoint_vis21)
        kp_coord_hw = tf.stack([kp_coord_h, kp_coord_w], 1)

        # determine size of crop (measure spatial extend of hw coords first)
        min_coord = tf.maximum(tf.reduce_min(kp_coord_hw, 0), 0.0)
        max_coord = tf.minimum(tf.reduce_max(kp_coord_hw, 0), image_size)

        # find out larger distance wrt the center of crop
        crop_size_best = 2 * tf.maximum(max_coord - crop_center, crop_center - min_coord)
        crop_size_best = tf.reduce_max(crop_size_best)
        crop_size_best = tf.minimum(tf.maximum(crop_size_best, 50.0), 500.0)

        # catch problem, when no valid kp available
        crop_size_best = tf.cond(tf.reduce_all(tf.is_finite(crop_size_best)), lambda: crop_size_best,
                                 lambda: tf.constant(200.0))
        crop_size_best.set_shape([])

        # calculate necessary scaling
        scale = tf.cast(crop_size, tf.float32) / crop_size_best
        scale = tf.minimum(tf.maximum(scale, 1.0), 10.0)
        scale *= crop_scale_noise
        crop_scale = scale

        if crop_offset_noise:
            noise = tf.truncated_normal([2], mean=0.0, stddev=crop_offset_noise_sigma)
            crop_center += noise

        # Crop image
        img_crop = crop_image_from_xy(tf.expand_dims(image, 0), crop_center, crop_size, scale)
        image_crop = tf.stack([img_crop[0, :, :, 0], img_crop[0, :, :, 1], img_crop[0, :, :, 2]], 2)

        # Modify uv21 coordinates
        crop_center_float = tf.cast(crop_center, tf.float32)
        keypoint_uv21_u = (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + crop_size // 2
        keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center_float[0]) * scale + crop_size // 2
        keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)

        # Modify camera intrinsics
        scale = tf.reshape(scale, [1, ])
        scale_matrix = tf.dynamic_stitch([[0], [1], [2],
                                          [3], [4], [5],
                                          [6], [7], [8]], [scale, [0.0], [0.0],
                                                           [0.0], scale, [0.0],
                                                           [0.0], [0.0], [1.0]])
        scale_matrix = tf.reshape(scale_matrix, [3, 3])

        crop_center_float = tf.cast(crop_center, tf.float32)
        trans1 = crop_center_float[0] * scale - crop_size // 2
        trans2 = crop_center_float[1] * scale - crop_size // 2
        trans1 = tf.reshape(trans1, [1, ])
        trans2 = tf.reshape(trans2, [1, ])
        trans_matrix = tf.dynamic_stitch([[0], [1], [2],
                                          [3], [4], [5],
                                          [6], [7], [8]], [[1.0], [0.0], -trans2,
                                                           [0.0], [1.0], -trans1,
                                                           [0.0], [0.0], [1.0]])
        trans_matrix = tf.reshape(trans_matrix, [3, 3])

        # 高斯分布
        keypoint_uv_heatmap = create_multiple_gaussian_map(keypoint_uv, image_size)
        """
        Segmentation masks available:
        左手：2,5,8,11,14
        右手：18,21,24,27,30
        """
        return image, keypoint_uv, keypoint_uv_heatmap
    def ReadTxtName(self, rootdir):
        lines = []
        with open(rootdir, 'r') as file_to_read:
            while True:
                line = file_to_read.readline()
                if not line:
                    break
                line = line.strip('\n')
                lines.append(line)
        return lines

if __name__ == '__main__':

    dataset_CMU = CMU()
    with tf.Session() as sess:

        for i in tqdm(range(dataset_CMU.sample_num)):
            image_crop, keypoint_uv21, keypoint_uv_heatmap = sess.run(dataset_CMU.get_batch_data)
            dataset_CMU.visualize_data(image_crop[0], keypoint_uv21[0],keypoint_uv_heatmap[0])
