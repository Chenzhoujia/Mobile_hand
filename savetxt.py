import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

centerx = 160
centery = 120
size = 24
color_ = (0, 255, 0)

cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    frame = scipy.misc.imresize(frame, (240, 320))

    image_raw1_crop = np.array(frame[centery - size: centery + size, centerx - size: centerx + size])
    image_raw1_crop = cv2.resize(image_raw1_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)

    first_point = (centerx - size, centery - size)
    last_point = (centerx + size, centery + size)
    cv2.rectangle(frame, first_point, last_point, color=color_, thickness=2)



    fig = plt.figure(1)
    plt.clf()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(frame)  # 第一个batch的维度 hand1(0~31) back1(32~63)
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.imshow(image_raw1_crop)  # 第一个batch的维度 hand1(0~31) back1(32~63)

    plt.pause(0.01)