# coding-utf8
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = None
    #hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
    #keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    #net.init(sess)

    # Feed image list through network
    while(True):
        # 获取相隔0.1s的两帧图片，做根据初始化或者网络输出结果，绘制矩形框，进行剪裁，reshape
        _,image_raw1 = cap.read()
        image_raw1 = scipy.misc.imresize(image_raw1, (240, 320))
        time.sleep(0.1)
        _,image_raw2 = cap.read()
        image_raw2 = scipy.misc.imresize(image_raw2, (240, 320))
            # 剪切 resize
        centerx = 160
        centery = 120
        size = 24
        image_raw1_crop = np.array(image_raw1[centery - size: centery + size, centerx - size: centerx + size])
        image_raw2_crop = np.array(image_raw2[centery - size: centery + size, centerx - size: centerx + size])
        image_raw1_crop = cv2.resize(image_raw1_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)
        image_raw2_crop = cv2.resize(image_raw2_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)
        first_point = (centerx-size, centery-size)
        last_point = (centerx+size, centery+size)
        cv2.rectangle(image_raw1, first_point, last_point, (0, 255, 0), 2)
        cv2.rectangle(image_raw2, first_point, last_point, (0, 255, 0), 2)


        image_v = np.expand_dims((image_raw1.astype('float') / 255.0) - 0.5, 0)



        # hand_scoremap_v, image_crop_v, scale_v, center_v,\
        # keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
        #                                                      keypoints_scoremap_tf, keypoint_coord3d_tf],
        #                                                     feed_dict={image_tf: image_v})

        # visualize
        fig = plt.figure(1)
        fig.clear()
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.imshow(image_raw1)
        ax2.imshow(image_raw2)
        ax3.imshow(image_raw1_crop)
        ax4.imshow(image_raw2_crop)

        #plt.show()
        plt.pause(0.01)
