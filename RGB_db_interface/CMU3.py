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
        for dat_i in tqdm(range(len(dat_all))):
            self.GAN_color_path.append(folderPath + dat_all[dat_i]['img_paths'])

        self.joint2D = np.loadtxt('/home/chen/Documents/Mobile_hand/RGB_db_interface/cache/joint2D.txt', delimiter=',')
        self.joint_pos = np.loadtxt('/home/chen/Documents/Mobile_hand/RGB_db_interface/cache/joint_pos.txt', delimiter=',')
        self.GAN_color_path = self.ReadTxtName('/home/chen/Documents/Mobile_hand/RGB_db_interface/cache/GAN_color_path.txt')
        self.sample_num = len(self.GAN_color_path)
        self.joint2D = np.reshape(self.joint2D,[-1,21,2])
        self.joint_pos = np.reshape(self.joint_pos,[-1,21,3])

        self.imagefilenames = tf.constant(self.GAN_color_path)


        # 创建训练数据集
        train_num = 140000

        dataset = tf.data.Dataset.from_tensor_slices((self.imagefilenames[:train_num], self.joint2D[:train_num], self.joint_pos[:train_num]))
        dataset = dataset.map(CMU._parse_function)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=320)
        self.dataset = dataset.batch(batchnum)
        #self.iterator = self.dataset.make_initializable_iterator() sess.run(dataset_RHD.iterator.initializer)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.get_batch_data = self.iterator.get_next()

        # 创建测试数据集
        dataset_eval = tf.data.Dataset.from_tensor_slices((self.imagefilenames[train_num:], self.joint2D[train_num:], self.joint_pos[train_num:]))
        dataset_eval = dataset_eval.map(CMU._parse_function)
        dataset_eval = dataset_eval.repeat()
        dataset_eval = dataset_eval.shuffle(buffer_size=320)
        self.dataset_eval = dataset_eval.batch(batchnum)
        #self.iterator = self.dataset.make_initializable_iterator() sess.run(dataset_RHD.iterator.initializer)
        self.iterator_eval = self.dataset_eval.make_one_shot_iterator()
        self.get_batch_data_eval = self.iterator_eval.get_next()


    @staticmethod
    def visualize_data(image_crop, keypoint_uv21, keypoint_uv_heatmap, keypoint_xyz21_normed):
        # get info from annotation dictionary

        image_crop = (image_crop + 0.5) * 255
        image_crop = image_crop.astype(np.int16)


        # visualize
        fig = plt.figure(1)
        plt.clf()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222, projection='3d')
        ax1.imshow(image_crop)
        #plot_hand(keypoint_uv21, ax1)
        ax1.scatter(keypoint_uv21[:, 0], keypoint_uv21[:, 1], s=10, c='k', marker='.')
        #ax1.scart(keypoint_uv21[:, 0], keypoint_uv21[:, 1], color=color, linewidth=1)
        plot_hand_3d(keypoint_xyz21_normed, ax2)
        ax2.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        ax2.set_xlim([-2, 2])
        ax2.set_ylim([-2, 2])
        ax2.set_zlim([-2, 2])

        ax3 = fig.add_subplot(223)
        ax3.imshow(np.sum(keypoint_uv_heatmap, axis=-1))  # 第一个batch的维度 hand1(0~31) back1(32~63)
        ax3.scatter(keypoint_uv21[:, 0], keypoint_uv21[:, 1], s=10, c='k', marker='.')
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        plt.savefig('/tmp/image/'+now+'.png')

    @staticmethod
    def _parse_function(imagefilename, keypoint_uv, keypoint_xyz):
        # 数据的基本处理
        image_size = (256, 256)
        image_string = tf.read_file(imagefilename)
        image_decoded = tf.image.decode_png(image_string)
        image_decoded = tf.image.resize_images(image_decoded, [256, 256], method=0)
        image_decoded.set_shape([image_size[0],image_size[0],3])
        image = tf.cast(image_decoded, tf.float32)
        image = image / 255.0 - 0.5


        # 高斯分布
        keypoint_uv_heatmap = create_multiple_gaussian_map(keypoint_uv, image_size)
        """
        Segmentation masks available:
        左手：2,5,8,11,14
        右手：18,21,24,27,30
        """
        return image, keypoint_uv, keypoint_uv_heatmap, keypoint_xyz
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

    plt.rcParams['figure.figsize'] = (20, 20)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    outpath = '/home/chen/Documents/Mobile_hand/RGB_db_interface/output_viz/'
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    # Input data paths
    folderPath = '/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/CMU3_hand143_panopticdb/'  # Put your local path here
    jsonPath = folderPath + 'hands_v143_14817.json'

    with open(jsonPath, 'r') as fid:
        dat_all = json.load(fid)
        dat_all = dat_all['root']

    dat = dat_all[0]  # Choose one element as an example;
    pts = np.array(dat['joint_self'])
    invalid = pts[:, 2] != 1

    imgPath = folderPath + dat['img_paths']
    # Plot annotations
    plt.clf()
    im = plt.imread(imgPath)
    plt.imshow(im)

    for p in range(pts.shape[0]):
        if pts[p, 2] != 0:
            plt.plot(pts[p, 0], pts[p, 1], 'r.')
            plt.text(pts[p, 0], pts[p, 1], '{0}'.format(p))
    for ie, e in enumerate(edges):
        if np.all(pts[e, 2] != 0):
            rgb = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0])
            plt.plot(pts[e, 0], pts[e, 1], color=rgb)
    plt.axis('off')
    plt.savefig(outpath + dat['img_paths'][5:-5] + '.jpg', bbox_inches='tight')


    dataset_CMU = CMU()
    # with tf.Session() as sess:
    #
    #     for i in tqdm(range(dataset_CMU.sample_num)):
    #         image_crop, keypoint_uv21, keypoint_uv_heatmap, keypoint_xyz21_normed = sess.run(dataset_CMU.get_batch_data_eval)
    #         dataset_CMU.visualize_data(image_crop[0], keypoint_uv21[0],keypoint_uv_heatmap[0], keypoint_xyz21_normed[0])
