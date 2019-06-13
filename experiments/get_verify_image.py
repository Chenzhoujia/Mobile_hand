import cv2
import scipy.misc
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
cap = cv2.VideoCapture(0)
# network input
# Feed image list through network
step = 0
centerx = 160
centery = 120
path = "/media/chen/4CBEA7F1BEA7D1AE/VR715/"
#
# for i in range(142):
#     image_raw1_crop = matplotlib.image.imread(path + str(step).zfill(5)+'_1.jpg')
#     image_raw2_crop = matplotlib.image.imread(path + str(step).zfill(5)+'_2.jpg')
#
#     fig = plt.figure(1)
#     fig.clear()
#     ax1 = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#
#     ax1.imshow(image_raw1_crop)
#     ax2.imshow(image_raw2_crop)
#     plt.pause(0.01)
#
#     step = step + 1
while (True):
    # 获取相隔0.1s的两帧图片，做根据初始化或者网络输出结果，绘制矩形框，进行剪裁，reshape
    _, image_raw1 = cap.read()
    image_raw1 = scipy.misc.imresize(image_raw1, (240, 320))
    # time.sleep(0.5)
    # _, image_raw2 = cap.read()
    # image_raw2 = scipy.misc.imresize(image_raw2, (240, 320))
    # 剪切 resize

    # size = 24
    # image_raw1_crop = np.array(image_raw1[centery - size: centery + size, centerx - size: centerx + size])
    # image_raw2_crop = np.array(image_raw2[centery - size: centery + size, centerx - size: centerx + size])
    # image_raw1_crop = cv2.resize(image_raw1_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)
    # image_raw2_crop = cv2.resize(image_raw2_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)
    # first_point = (centerx - size, centery - size)
    # last_point = (centerx + size, centery + size)
    # cv2.rectangle(image_raw1, first_point, last_point, (0, 255, 0), 2)
    # cv2.rectangle(image_raw2, first_point, last_point, (0, 255, 0), 2)
    #
    # # visualize
    #
    #
    # fig = plt.figure(1)
    # fig.clear()
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax4 = fig.add_subplot(224)
    #
    # ax1.imshow(image_raw1)
    # ax2.imshow(image_raw2)
    # ax3.imshow(image_raw1_crop)
    # ax4.imshow(image_raw2_crop)
    # plt.savefig(path + ".jpg")
    # plt.pause(0.01)

    matplotlib.image.imsave(path + str(step).zfill(5)+'.jpg', image_raw1)
    # matplotlib.image.imsave(path + str(step).zfill(5)+'_2.jpg', image_raw2_crop)

    step = step + 1