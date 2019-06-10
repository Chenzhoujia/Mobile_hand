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
        self.imagefilenames = tf.constant(imagefilenames)

        maskfilenames = BaseDataset.listdir(self.path+"/training/mask")
        self.maskfilenames = tf.constant(maskfilenames)

        assert self.example_num == len(anno_all), '标签和样本数量不一致'

        # 创建背景数据集
        dataset_back = tf.data.Dataset.from_tensor_slices((self.imagefilenames, self.maskfilenames, self.allxyz, self.alluv, self.allk))
        dataset_back = dataset_back.map(RHD._parse_function_background)
        dataset_back = dataset_back.repeat()
        dataset_back = dataset_back.shuffle(buffer_size=320)
        self.dataset_back = dataset_back.batch(batchnum)
        self.iterator_back = self.dataset_back.make_one_shot_iterator()
        self.get_batch_back_data = self.iterator_back.get_next()

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
    def visualize_data(image, xyz, uv, uv_vis, k, num_px_left_hand, num_px_right_hand, scoremap,
                       image_crop_tip, crop_size_best,  image_nohand, scoremap_1, scoremap_2):
        # get info from annotation dictionary

        image = (image + 0.5) * 255
        image = image.astype(np.int16)

        image_crop_tip = (image_crop_tip + 0.5) * 255
        image_crop_tip = image_crop_tip.astype(np.int16)

        image_nohand = (image_nohand + 0.5) * 255
        image_nohand = image_nohand.astype(np.int16)


        kp_coord_uv = uv[:, :2]  # u, v coordinates of 42 hand keypoints, pixel
        kp_visible =(uv_vis == 1)  # visibility of the keypoints, boolean
        kp_coord_xyz = xyz  # x, y, z coordinates of the keypoints, in meters
        camera_intrinsic_matrix = k  # matrix containing intrinsic parameters
        # Project world coordinates into the camera frame
        kp_coord_uv_proj = np.matmul(kp_coord_xyz, np.transpose(camera_intrinsic_matrix))
        kp_coord_uv_proj = kp_coord_uv_proj[:, :2] / kp_coord_uv_proj[:, 2:]

        # Visualize data
        fig = plt.figure(1)
        ax1 = fig.add_subplot('331')
        ax1.set_title("left_hand:"+str(num_px_left_hand)+"right_hand:"+str(num_px_right_hand))
        ax2 = fig.add_subplot('332')
        ax3 = fig.add_subplot('333')
        ax4 = fig.add_subplot('334', projection='3d')
        ax5 = fig.add_subplot('335')

        ax5.imshow(image_crop_tip)
        ax5.set_title(str(crop_size_best[0]))
        ax6 = fig.add_subplot('336')
        ax6.imshow(image_nohand)
        ax1.imshow(image)
        ax1.scatter(kp_coord_uv[kp_visible, 0], kp_coord_uv[kp_visible, 1], marker='o', color='blue', s=5)
        #ax2.scatter(kp_coord_uv_proj[:, 0], kp_coord_uv_proj[:, 1], marker='x', color='red', s=5)
        ax2.imshow(np.sum(scoremap_1, axis=-1))
        sum = np.sum(np.sum(scoremap_1, axis=0), axis=0)
        ax2.set_title(str(sum))
        ax3.imshow(np.sum(scoremap_2, axis=-1))
        sum = np.sum(np.sum(scoremap_2, axis=0), axis=0)
        ax3.set_title(str(sum))
        ax4.plot([0,crop_size_best[1]], [0,crop_size_best[2]], [0,crop_size_best[3]])
        ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        ax4.set_xlabel('x')
        ax4.set_xlim((-1, 1))
        ax4.set_ylim((-1, 1))
        ax4.set_zlim((-1, 1))
        ax4.set_ylabel('y')
        ax4.set_zlabel('z')
        ax8 = fig.add_subplot('338')
        ax9 = fig.add_subplot('339')
        plt.show()

    @staticmethod
    def _parse_function_background(imagefilename, maskfilename, keypoint_xyz, keypoint_uv, k):
        # 数据的基本处理
        image_size = (320, 320)
        image_string = tf.read_file(imagefilename)
        image_decoded = tf.image.decode_png(image_string)
        image_decoded.set_shape([image_size[0], image_size[0], 3])
        image = tf.cast(image_decoded, tf.float32)
        image = image / 255.0 - 0.5

        mask_string = tf.read_file(maskfilename)
        mask_decoded = tf.image.decode_png(mask_string)
        hand_parts_mask = tf.cast(mask_decoded, tf.int32)
        hand_parts_mask.set_shape([image_size[0], image_size[0], 1])

        # general parameters
        crop_size = 32
        hue_aug = True
        hue_aug_max = 0.1

        if hue_aug:
            image = tf.image.random_hue(image, hue_aug_max)

        # 数据的高级处理
        #（3）寻找no hand图片

        one_map, zero_map = tf.ones_like(hand_parts_mask), tf.zeros_like(hand_parts_mask)
        no_hand_mask = tf.less(hand_parts_mask, one_map*2)
        no_hand_mask = tf.stack([no_hand_mask[:, :, 0], no_hand_mask[:, :, 0], no_hand_mask[:, :, 0]], 2)

        rgb_mean = tf.reduce_mean(image, axis=0)
        rgb_mean = tf.reduce_mean(rgb_mean, axis=0)

        back_image = tf.ones_like(image)
        back_image = tf.stack([back_image[:, :, 0]*rgb_mean[0], back_image[:, :, 1]*rgb_mean[1], back_image[:, :, 2]*rgb_mean[2]], 2)
        image_nohand = tf.where(no_hand_mask, image, back_image)
        image_nohand1 = tf.random_crop(image_nohand, [crop_size, crop_size, 3])
        image_nohand2 = tf.random_crop(image_nohand, [crop_size, crop_size, 3])

        # return image, keypoint_xyz21, keypoint_uv21, scoremap,
        #  keypoint_vis21, k, num_px_left_hand, num_px_right_hand, \
        #        image_crop_comb, hand_motion, image_crop_comb2, scoremap1, scoremap2
        return image, tf.zeros([21,3], dtype=tf.float32), tf.zeros([21,2], dtype=tf.float32), tf.zeros([320,320,5], dtype=tf.float32),\
               tf.zeros([21], dtype=tf.bool), tf.zeros([3,3], dtype=tf.float32), tf.zeros([], dtype=tf.int32), tf.zeros([], dtype=tf.int32),\
               image_nohand1, tf.zeros([4], dtype=tf.float32), image_nohand2, tf.zeros([32,32,5], dtype=tf.float32),tf.zeros([32,32,5], dtype=tf.float32)

    @staticmethod
    def _parse_function(imagefilename, maskfilename, keypoint_xyz, keypoint_uv, k):
        # 数据的基本处理
        image_size = (320, 320)
        image_string = tf.read_file(imagefilename)
        image_decoded = tf.image.decode_png(image_string)
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
        hand_crop = False
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
        keypoint_xyz21_normed = kp_coord_xyz21_rel / index_root_bone_length  # normalized by length of 12->11

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
            keypoint_uv21 = tf.stack([keypoint_uv21_u, keypoint_uv21_v], 1)
            keypoint_uv21 = keypoint_uv21

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

        """ DEPENDENT DATA ITEMS: Scoremap from the SUBSET of 21 keypoints"""
        # create scoremaps from the subset of 2D annoataion
        keypoint_hw21 = tf.stack([keypoint_uv21[:, 1], keypoint_uv21[:, 0]], -1)

        scoremap_size = image_size

        if hand_crop:
            scoremap_size = (crop_size, crop_size)

        scoremap = BaseDataset.create_multiple_gaussian_map(keypoint_hw21,
                                                             scoremap_size,
                                                             sigma,
                                                             valid_vec=keypoint_vis21)

        # if scoremap_dropout:
        #     scoremap = tf.nn.dropout(scoremap, scoremap_dropout_prob,
        #                              noise_shape=[1, 1, 21])
        #     scoremap *= scoremap_dropout_prob
        scoremap = tf.concat([tf.expand_dims(scoremap[:,:,1],-1),tf.expand_dims(scoremap[:,:,5],-1)
                              ,tf.expand_dims(scoremap[:,:,9],-1),tf.expand_dims(scoremap[:,:,13],-1)
                              ,tf.expand_dims(scoremap[:, :, 17], -1)], 2)
        image_crop_comb, hand_motion, image_crop_comb2,  scoremap1, scoremap2\
            = RHD._parse_function_furtner(image, keypoint_uv21, hand_parts_mask, scoremap)

        return image, keypoint_xyz21, keypoint_uv21, scoremap, keypoint_vis21, k, num_px_left_hand, num_px_right_hand, \
               image_crop_comb, hand_motion, image_crop_comb2, scoremap1, scoremap2


    """
    Keypoints available:
    0: left wrist, 1-4: left thumb [tip to palm], 5-8: left index, ..., 17-20: left pinky,
    Segmentation masks available:
    0: background, 1: person, 
    2-4: left thumb [tip to palm], 5-7: left index, ..., 14-16: left pinky, 17: palm, 
    18-20: right thumb, ..., right palm: 33
    """
    @staticmethod
    def _parse_function_furtner(image, keypoint_uv21, hand_parts_mask, scoremap=None):

        crop_size = 32

        #（1）计算切割大小
        #舍弃3D坐标，因为我们只需要xy轴的位移百分比和z轴尺度的变化百分比，网络的输入与相机矩阵没有关系，网络的输出也只是像素图像上的
        #使用编号为5的节点 crop，按照56关节之间的长度的两倍计算crop大小（大概(10~40)个px），以指尖为中心
        crop_center = keypoint_uv21[5, ::-1]
        crop_center_wing = keypoint_uv21[6, ::-1]
            # catch problem, when no valid kp available (happens almost never)
        crop_center = tf.cond(tf.reduce_all(tf.is_finite(crop_center)), lambda: crop_center,
                              lambda: tf.constant([0.0, 0.0]))
        crop_center.set_shape([2, ])
        crop_center_wing = tf.cond(tf.reduce_all(tf.is_finite(crop_center_wing)), lambda: crop_center_wing,
                              lambda: tf.constant([0.0, 0.0]))
        crop_center_wing.set_shape([2, ])
        crop_size_best = tf.abs(2 * (crop_center - crop_center_wing))
        crop_size_best = tf.reduce_max(crop_size_best)
        crop_size_best = tf.minimum(tf.maximum(crop_size_best, 10.0), 50.0)
            # catch problem, when no valid kp available
        crop_size_best = tf.cond(tf.reduce_all(tf.is_finite(crop_size_best)), lambda: crop_size_best,
                                 lambda: tf.constant(20.0))
        crop_size_best.set_shape([])

        #（2）判断关节点是否被遮挡，不确定这是否有意义，只能用作选择model的依据。其次必须借助maskh和visable

        #（3）寻找no hand图片

        one_map, zero_map = tf.ones_like(hand_parts_mask), tf.zeros_like(hand_parts_mask)
        no_hand_mask = tf.less(hand_parts_mask, one_map*2)
        no_hand_mask = tf.stack([no_hand_mask[:, :, 0], no_hand_mask[:, :, 0], no_hand_mask[:, :, 0]], 2)

        rgb_mean = tf.reduce_mean(image, axis=0)
        rgb_mean = tf.reduce_mean(rgb_mean, axis=0)

        back_image = tf.ones_like(image)
        back_image = tf.stack([back_image[:, :, 0]*rgb_mean[0], back_image[:, :, 1]*rgb_mean[1], back_image[:, :, 2]*rgb_mean[2]], 2)
        image_nohand = tf.where(no_hand_mask, image, back_image)
        image_nohand = tf.random_crop(image_nohand, [crop_size*3, crop_size*3, 3])



        # （4）剪切 按照crop_size_best截取像素 缩放到 crop_size
        scale = tf.cast(crop_size, tf.float32) / crop_size_best
        img_crop = crop_image_from_xy(tf.expand_dims(image, 0), crop_center, crop_size*3, scale)
        image_crop = tf.stack([img_crop[0, :, :, 0], img_crop[0, :, :, 1], img_crop[0, :, :, 2]], 2)

        hand_parts_mask_crop = crop_image_from_xy(tf.expand_dims(hand_parts_mask, 0), crop_center, crop_size*3, scale)
        hand_parts_mask_crop = tf.stack([hand_parts_mask_crop[0, :, :, 0], hand_parts_mask_crop[0, :, :, 0], hand_parts_mask_crop[0, :, :, 0]], 2)

        scoremap = crop_image_from_xy(tf.expand_dims(scoremap, 0), crop_center, crop_size*3, scale)
        scoremap = tf.stack([scoremap[0, :, :, 0], scoremap[0, :, :, 1], scoremap[0, :, :, 2],
                             scoremap[0, :, :, 3], scoremap[0, :, :, 4]], 2)
        #制作5个指尖的mask，对scoremap的5个维度进行过滤
        ones_mask = tf.ones_like(hand_parts_mask_crop[:, :, 0])
        zeros_mask = tf.zeros_like(scoremap[:, :, 1])

        finger_mask = tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 2) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 18) | \
                      tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 3) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 19)
        scoremap_mask1 = tf.where(finger_mask, scoremap[:, :, 0], zeros_mask)

        finger_mask = tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 5) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 21) | \
                      tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 6) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 22)
        scoremap_mask2 = tf.where(finger_mask, scoremap[:, :, 1], zeros_mask)

        finger_mask = tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 8) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 24) | \
                      tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 9) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 25)
        scoremap_mask3 = tf.where(finger_mask, scoremap[:, :, 2], zeros_mask)

        finger_mask = tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 11) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 27) | \
                      tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 12) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 28)
        scoremap_mask4 = tf.where(finger_mask, scoremap[:, :, 3], zeros_mask)

        finger_mask = tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 14) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 30) | \
                      tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 15) | tf.equal(hand_parts_mask_crop[:, :, 0], ones_mask * 31)
        scoremap_mask5 = tf.where(finger_mask, scoremap[:, :, 4], zeros_mask)


        scoremap = tf.stack([scoremap_mask1, scoremap_mask2, scoremap_mask3, scoremap_mask4, scoremap_mask5], 2)
        # (5) image_crop hand_parts_mask_crop 是一个3倍于原图大小的剪切图片 二者需要做相同的旋转、平移、缩放操作
        #随机生成6个百分比，对原始大图image，hand_parts_mask进行变换，重新执行（4）（5）
        def tf_image_translate(images, tx, ty, interpolation='NEAREST'):
            # got these parameters from solving the equations for pixel translations
            # on https://www.tensorflow.org/api_docs/python/tf/contrib/image/transform
            transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
            return tf.contrib.image.transform(images, transforms, interpolation)
        hand_motion = tf.truncated_normal([4], mean=0.0, stddev=0.25)# -0.5 ~ +0.5 [r, x, y, z]

        image_crop2 = tf.contrib.image.rotate(image_crop, np.pi*hand_motion[0]*0.1, interpolation='NEAREST', name=None)
        image_crop2 = tf_image_translate(image_crop2, tx=crop_size*hand_motion[1], ty=crop_size*hand_motion[2])

        hand_parts_mask_crop2 = tf.contrib.image.rotate(hand_parts_mask_crop, np.pi*hand_motion[0]*0.1, interpolation='NEAREST', name=None)
        hand_parts_mask_crop2 = tf_image_translate(hand_parts_mask_crop2, tx=crop_size*hand_motion[1], ty=crop_size*hand_motion[2])

        scoremap2 = tf.contrib.image.rotate(scoremap, np.pi*hand_motion[0]*0.1, interpolation='NEAREST', name=None)
        scoremap2 = tf_image_translate(scoremap2, tx=crop_size*hand_motion[1], ty=crop_size*hand_motion[2])

        #（6）根据hand_parts_mask_crop， image_nohand， image_crop， 填充不变背景图片
        no_hand_mask = tf.ones_like(hand_parts_mask_crop)
        no_hand_mask = tf.less(hand_parts_mask_crop, no_hand_mask * 2) #小于2的背景和人体（no hand）像素为true
        image_crop_comb = tf.where(no_hand_mask, image_nohand, image_crop)
        no_hand_mask2 = tf.ones_like(hand_parts_mask_crop2)
        no_hand_mask2 = tf.less(hand_parts_mask_crop2, no_hand_mask2 * 2) #小于2的背景和人体（no hand）像素为true
        image_crop_comb2 = tf.where(no_hand_mask2, image_nohand, image_crop2)

        #(7)随机切割
        center_alph = tf.truncated_normal([2], mean=0.0, stddev=0.5)# -1 ~ +1 [r, x, y, z]
        tx = crop_size * hand_motion[1]
        ty = crop_size * hand_motion[2]

        center_center = [crop_size * 3//2+(1+center_alph[1])*ty//2, crop_size * 3//2+(1+center_alph[0])*tx//2]

        image_crop2_comb = crop_image_from_xy(tf.expand_dims(image_crop_comb, 0), center_center, crop_size)
        image_crop2_comb = tf.stack([image_crop2_comb[0, :, :, 0], image_crop2_comb[0, :, :, 1], image_crop2_comb[0, :, :, 2]], 2)

        #缩放切割 切割成crop_size*(1+hand_motion[3]*0.3)大小，然后resize缩放，从而模拟z轴变化
        image_crop2_comb2 = crop_image_from_xy(tf.expand_dims(image_crop_comb2, 0), center_center,crop_size, crop_size/(crop_size*(1+hand_motion[3]*0.01)))
        image_crop2_comb2 = tf.stack([image_crop2_comb2[0, :, :, 0], image_crop2_comb2[0, :, :, 1], image_crop2_comb2[0, :, :, 2]], 2)

        scoremap = crop_image_from_xy(tf.expand_dims(scoremap, 0), center_center, crop_size)
        scoremap = tf.stack([scoremap[0, :, :, 0], scoremap[0, :, :, 1], scoremap[0, :, :, 2],
                             scoremap[0, :, :, 3], scoremap[0, :, :, 4]], 2)
        scoremap2 = crop_image_from_xy(tf.expand_dims(scoremap2, 0), center_center, crop_size, crop_size/(crop_size*(1+hand_motion[3]*0.01)))
        scoremap2 = tf.stack([scoremap2[0, :, :, 0], scoremap2[0, :, :, 1], scoremap2[0, :, :, 2],
                             scoremap2[0, :, :, 3], scoremap2[0, :, :, 4]], 2)

        # hand_parts_mask_crop = crop_image_from_xy(tf.expand_dims(hand_parts_mask_crop, 0), center_center, crop_size)
        # hand_parts_mask_crop = tf.stack([hand_parts_mask_crop[0, :, :, 0], hand_parts_mask_crop[0, :, :, 0], hand_parts_mask_crop[0, :, :, 0]], 2)
        # hand_parts_mask_crop2 = crop_image_from_xy(tf.expand_dims(hand_parts_mask_crop2, 0), center_center, crop_size, crop_size/(crop_size*(1+hand_motion[3]*0.01)))
        # hand_parts_mask_crop2 = tf.stack([hand_parts_mask_crop2[0, :, :, 0], hand_parts_mask_crop2[0, :, :, 0], hand_parts_mask_crop2[0, :, :, 0]], 2)

        def scoremap_filter(scoremap):
            zeros_mask = tf.zeros_like(scoremap[:,:,0])
            scoremap_mask1 = scoremap[:,:,0]
            scoremap_mask2 = scoremap[:,:,1]
            scoremap_mask3 = scoremap[:,:,2]
            scoremap_mask4 = scoremap[:,:,3]
            scoremap_mask5 = scoremap[:,:,4]

            scoremap_mask1 = tf.cond(tf.less(x=tf.reduce_sum(scoremap_mask1),y=tf.constant(10.0)), true_fn=lambda: zeros_mask,
                                     false_fn=lambda: scoremap_mask1)
            scoremap_mask2 = tf.cond(tf.less(x=tf.reduce_sum(scoremap_mask2),y=tf.constant(10.0)), true_fn=lambda: zeros_mask,
                                     false_fn=lambda: scoremap_mask2)
            scoremap_mask3 = tf.cond(tf.less(x=tf.reduce_sum(scoremap_mask3),y=tf.constant(10.0)), true_fn=lambda: zeros_mask,
                                     false_fn=lambda: scoremap_mask3)
            scoremap_mask4 = tf.cond(tf.less(x=tf.reduce_sum(scoremap_mask4),y=tf.constant(10.0)), true_fn=lambda: zeros_mask,
                                     false_fn=lambda: scoremap_mask4)
            scoremap_mask5 = tf.cond(tf.less(x=tf.reduce_sum(scoremap_mask5),y=tf.constant(10.0)), true_fn=lambda: zeros_mask,
                                     false_fn=lambda: scoremap_mask5)

            return tf.stack([scoremap_mask1, scoremap_mask2, scoremap_mask3, scoremap_mask4, scoremap_mask5], 2)

        scoremap = scoremap_filter(scoremap)
        scoremap2 = scoremap_filter(scoremap2)
        # 当热度图中任意一个为空时，将handmotion置0
        hand_motion = tf.cond(tf.less(x=tf.reduce_sum(scoremap), y=tf.constant(10.0)), true_fn=lambda: tf.zeros_like(hand_motion),
                      false_fn=lambda: hand_motion)
        hand_motion = tf.cond(tf.less(x=tf.reduce_sum(scoremap2), y=tf.constant(10.0)), true_fn=lambda: tf.zeros_like(hand_motion),
                      false_fn=lambda: hand_motion)


        return image_crop2_comb, hand_motion, image_crop2_comb2, scoremap, scoremap2

dataset_RHD = RHD()
with tf.Session() as sess:

    for i in tqdm(range(dataset_RHD.example_num)):
        image, keypoint_xyz, keypoint_uv, scoremap, keypoint_vis, k, num_px_left_hand, num_px_right_hand, \
        image_crop_comb, hand_motion, image_crop_comb2,  scoremap1, scoremap2 = sess.run(dataset_RHD.get_batch_data)

        RHD.visualize_data(image[0], keypoint_xyz[0], keypoint_uv[0], keypoint_vis[0], k[0], num_px_left_hand[0], num_px_right_hand[0], scoremap[0],
                           image_crop_comb[0], hand_motion[0], image_crop_comb2[0], scoremap1[0], scoremap2[0])

        # image, keypoint_xyz, keypoint_uv, scoremap, keypoint_vis, k, num_px_left_hand, num_px_right_hand, \
        # image_crop_comb, hand_motion, image_crop_comb2,  scoremap1, scoremap2 = sess.run(dataset_RHD.get_batch_back_data)
        #
        # RHD.visualize_data(image[0], keypoint_xyz[0], keypoint_uv[0], keypoint_vis[0], k[0], num_px_left_hand[0], num_px_right_hand[0], scoremap[0],
        #                    image_crop_comb[0], hand_motion[0], image_crop_comb2[0], scoremap1[0], scoremap2[0])
