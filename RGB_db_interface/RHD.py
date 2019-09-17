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
def plot_hand(coords_hw, axis, color_fixed=None, linewidth='1'):
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
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
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
    bones = [((0, 4), colors[0, :]),
             ((4, 3), colors[1, :]),
             ((3, 2), colors[2, :]),
             ((2, 1), colors[3, :]),

             ((0, 8), colors[4, :]),
             ((8, 7), colors[5, :]),
             ((7, 6), colors[6, :]),
             ((6, 5), colors[7, :]),

             ((0, 12), colors[8, :]),
             ((12, 11), colors[9, :]),
             ((11, 10), colors[10, :]),
             ((10, 9), colors[11, :]),

             ((0, 16), colors[12, :]),
             ((16, 15), colors[13, :]),
             ((15, 14), colors[14, :]),
             ((14, 13), colors[15, :]),

             ((0, 20), colors[16, :]),
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
class RHD(BaseDataset):
    def __init__(self, path="/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/ICCV2017/RHD_published_v2", batchnum = 4):
        super(RHD, self).__init__(path)

        # 将标签加载到内存中
        with open(self.path+"/training/anno_training.pickle", 'rb') as fi:
            anno_all = pickle.load(fi)
        self.allxyz = np.zeros(shape=[len(anno_all), 42, 3], dtype=np.float32)
        self.alluv = np.zeros(shape=[len(anno_all), 42, 3], dtype=np.float32)
        self.allk = np.zeros(shape=[len(anno_all), 3, 3], dtype=np.float32)
        for label_i in range(len(anno_all)):
            self.allxyz[label_i,:,:] = anno_all[label_i]['xyz']
            self.alluv[label_i, :, :] = anno_all[label_i]['uv_vis']
            self.allk[label_i, :, :] = anno_all[label_i]['K']

        # 将图像文件名列表加载到内存中
        imagefilenames = BaseDataset.listdir(self.path+"/training/color")
        self.example_num = len(imagefilenames)
        maskfilenames = BaseDataset.listdir(self.path+"/training/mask")

        assert self.example_num == len(anno_all), '标签和样本数量不一致'

        self.maskfilenames = tf.constant(maskfilenames)
        self.imagefilenames = tf.constant(imagefilenames)
        # 创建正经数据集
        dataset = tf.data.Dataset.from_tensor_slices((self.imagefilenames, self.maskfilenames, self.allxyz, self.alluv, self.allk))
        dataset = dataset.map(RHD._parse_function)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=320)
        self.dataset = dataset.batch(batchnum)
        #self.iterator = self.dataset.make_initializable_iterator() sess.run(dataset_RHD.iterator.initializer)
        self.iterator = self.dataset.make_one_shot_iterator()
        self.get_batch_data = self.iterator.get_next()


    @staticmethod
    def visualize_data(image, image_crop, keypoint_uv21, keypoint_xyz21_normed):
        # get info from annotation dictionary

        image = (image + 0.5) * 255
        image = image.astype(np.int16)

        image_crop = (image_crop + 0.5) * 255
        image_crop = image_crop.astype(np.int16)


        # visualize
        fig = plt.figure(1)
        plt.clf()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224, projection='3d')
        ax1.imshow(image)
        ax2.imshow(image_crop)
        plot_hand(keypoint_uv21, ax1)
        ax1.scatter(keypoint_uv21[:, 0], keypoint_uv21[:, 1], s=10, c='k', marker='.')
        #ax1.scart(keypoint_uv21[:, 0], keypoint_uv21[:, 1], color=color, linewidth=1)
        plot_hand(keypoint_uv21, ax3)
        ax3.set_xlim([0,300])
        ax3.set_ylim([0,300])
        plot_hand_3d(keypoint_xyz21_normed, ax4)
        ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        ax4.set_xlim([-3, 3])
        ax4.set_ylim([-3, 3])
        ax4.set_zlim([-3, 3])

        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        plt.savefig('/tmp/image/'+now+'.png')

    @staticmethod
    def _parse_function(imagefilename, maskfilename, keypoint_xyz, keypoint_uv, k):
        # 数据的基本处理
        image_size = (320, 320)
        image_string = tf.read_file(imagefilename)
        image_decoded = tf.image.decode_png(image_string)
        image_decoded = tf.image.resize_images(image_decoded, [320, 320], method=0)
        image_decoded.set_shape([image_size[0],image_size[0],3])
        image = tf.cast(image_decoded, tf.float32)
        image = image / 255.0 - 0.5

        mask_string = tf.read_file(maskfilename)
        mask_decoded = tf.image.decode_png(mask_string)
        hand_parts_mask = tf.cast(mask_decoded, tf.int32)
        hand_parts_mask.set_shape([image_size[0],image_size[0],1])

        keypoint_vis = tf.cast(keypoint_uv[:, 2], tf.bool)
        keypoint_uv = keypoint_uv[:, :2]

        # general parameters
        coord_uv_noise = False
        coord_uv_noise_sigma = 1.0  # std dev in px of noise on the uv coordinates
        crop_center_noise = True
        crop_center_noise_sigma = 20.0  # std dev in px: this moves what is in the "center", but the crop always contains all keypoints
        crop_offset_noise = False
        crop_offset_noise_sigma = 10.0  # translates the crop after size calculation (this can move keypoints outside)
        crop_scale_noise_ = False
        crop_size = 192
        hand_crop = True
        hue_aug = False
        hue_aug_max = 0.1
        sigma = 6.0
        use_wrist_coord = False

        # 使用掌心代替手腕
        if not use_wrist_coord:
            palm_coord_l = tf.expand_dims(0.5*(keypoint_xyz[0, :] + keypoint_xyz[12, :]), 0)
            palm_coord_r = tf.expand_dims(0.5*(keypoint_xyz[21, :] + keypoint_xyz[33, :]), 0)
            keypoint_xyz = tf.concat([palm_coord_l, keypoint_xyz[1:21, :], palm_coord_r, keypoint_xyz[-20:, :]], 0)

        if not use_wrist_coord:
            palm_coord_uv_l = tf.expand_dims(0.5*(keypoint_uv[0, :] + keypoint_uv[12, :]), 0)
            palm_coord_uv_r = tf.expand_dims(0.5*(keypoint_uv[21, :] + keypoint_uv[33, :]), 0)
            keypoint_uv = tf.concat([palm_coord_uv_l, keypoint_uv[1:21, :], palm_coord_uv_r, keypoint_uv[-20:, :]], 0)

        if coord_uv_noise:
            noise = tf.truncated_normal([42, 2], mean=0.0, stddev=coord_uv_noise_sigma)
            keypoint_uv += noise

        if hue_aug:
            image = tf.image.random_hue(image, hue_aug_max)

        # calculate palm visibility
        if not use_wrist_coord:
            palm_vis_l = tf.expand_dims(tf.logical_or(keypoint_vis[0], keypoint_vis[12]), 0)
            palm_vis_r = tf.expand_dims(tf.logical_or(keypoint_vis[21], keypoint_vis[33]), 0)
            keypoint_vis = tf.concat([palm_vis_l, keypoint_vis[1:21], palm_vis_r, keypoint_vis[-20:]], 0)

        # 数据的高级处理
        # figure out dominant hand by analysis of the segmentation mask
        one_map, zero_map = tf.ones_like(hand_parts_mask), tf.zeros_like(hand_parts_mask)
        cond_l = tf.logical_and(tf.greater(hand_parts_mask, one_map), tf.less(hand_parts_mask, one_map*18))
        cond_r = tf.greater(hand_parts_mask, one_map*17)
        hand_map_l = tf.where(cond_l, one_map, zero_map)
        hand_map_r = tf.where(cond_r, one_map, zero_map)
        num_px_left_hand = tf.reduce_sum(hand_map_l)
        num_px_right_hand = tf.reduce_sum(hand_map_r)

        # We only deal with the more prominent hand for each frame and discard the second set of keypoints
        kp_coord_xyz_left = keypoint_xyz[:21, :]
        kp_coord_xyz_right = keypoint_xyz[-21:, :]

        cond_left = tf.logical_and(tf.cast(tf.ones_like(kp_coord_xyz_left), tf.bool), tf.greater(num_px_left_hand, num_px_right_hand))
        kp_coord_xyz21 = tf.where(cond_left, kp_coord_xyz_left, kp_coord_xyz_right)

        hand_side = tf.where(tf.greater(num_px_left_hand, num_px_right_hand),
                             tf.constant(0, dtype=tf.int32),
                             tf.constant(1, dtype=tf.int32))  # left hand = 0; right hand = 1
        hand_side = tf.one_hot(hand_side, depth=2, on_value=1.0, off_value=0.0, dtype=tf.float32)

        keypoint_xyz21 = kp_coord_xyz21
        #对xyz标签进行标准化
        # make coords relative to root joint
        kp_coord_xyz_root = kp_coord_xyz21[0, :] # this is the palm coord
        kp_coord_xyz21_rel = kp_coord_xyz21 - kp_coord_xyz_root  # relative coords in metric coords
        index_root_bone_length = tf.sqrt(tf.reduce_sum(tf.square(kp_coord_xyz21_rel[12, :] - kp_coord_xyz21_rel[11, :])))
        keypoint_xyz21_normed = kp_coord_xyz21_rel / (index_root_bone_length+0.0001)  # normalized by length of 12->11

        # calculate local coordinates
        kp_coord_xyz21_local = bone_rel_trafo(keypoint_xyz21_normed)
        kp_coord_xyz21_local = tf.squeeze(kp_coord_xyz21_local)
        keypoint_xyz21_local = kp_coord_xyz21_local

        # calculate viewpoint and coords in canonical coordinates
        kp_coord_xyz21_rel_can, rot_mat = canonical_trafo(keypoint_xyz21_normed)
        kp_coord_xyz21_rel_can, rot_mat = tf.squeeze(kp_coord_xyz21_rel_can), tf.squeeze(rot_mat)
        kp_coord_xyz21_rel_can = flip_right_hand(kp_coord_xyz21_rel_can, tf.logical_not(cond_left))
        keypoint_xyz21_can = kp_coord_xyz21_rel_can
        rot_mat = tf.matrix_inverse(rot_mat)

        # Set of 21 for visibility
        keypoint_vis_left = keypoint_vis[:21]
        keypoint_vis_right = keypoint_vis[-21:]
        keypoint_vis21 = tf.where(cond_left[:, 0], keypoint_vis_left, keypoint_vis_right)
        keypoint_vis21 = keypoint_vis21

        # Set of 21 for UV coordinates
        keypoint_uv_left = keypoint_uv[:21, :]
        keypoint_uv_right = keypoint_uv[-21:, :]
        keypoint_uv21 = tf.where(cond_left[:, :2], keypoint_uv_left, keypoint_uv_right)

        """ DEPENDENT DATA ITEMS: HAND CROP """
        if hand_crop:
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
            crop_size_best = 2*tf.maximum(max_coord - crop_center, crop_center - min_coord)
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
            image_crop = tf.stack([img_crop[0,:,:,0], img_crop[0,:,:,1],img_crop[0,:,:,2]],2)

            # Modify uv21 coordinates
            crop_center_float = tf.cast(crop_center, tf.float32)
            keypoint_uv21_u = (keypoint_uv21[:, 0] - crop_center_float[1]) * scale + crop_size // 2
            keypoint_uv21_v = (keypoint_uv21[:, 1] - crop_center_float[0]) * scale + crop_size // 2
            # keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)

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

            k = tf.matmul(trans_matrix, tf.matmul(scale_matrix, k))

        # 统计指尖像素数量，指导高斯分布的面积
        """
        Segmentation masks available:
        左手：2,5,8,11,14
        右手：18,21,24,27,30
        """
        return image, image_crop, keypoint_uv21, keypoint_xyz21_normed

if __name__ == '__main__':
    dataset_RHD = RHD()
    with tf.Session() as sess:

        for i in tqdm(range(dataset_RHD.example_num)):
            image, image_crop, keypoint_uv21, keypoint_xyz21_normed = sess.run(dataset_RHD.get_batch_data)
            RHD.visualize_data(image[0], image_crop[0], keypoint_uv21[0], keypoint_xyz21_normed[0])
