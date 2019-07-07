import tensorflow as tf
import argparse
import cv2
from tqdm import tqdm
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


def load_graph(frozen_graph_filename):
    # 加载protobug文件，并反序列化成graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph_:
        # 将读出来的graph_def导入到当前的Graph中
        # 为了避免多个图之间的明明冲突，增加一个前缀
        tf.import_graph_def(graph_def, name="prefix")

    return graph_


if __name__ == '__main__':

    # 允许用户传入文件名作为参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()

    # 从pb文件中读取图结构
    graph = load_graph(args.frozen_model_filename)

    # 列举所有的操作
    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('prefix/GPU_0/input_image:0')
    y = graph.get_tensor_by_name('prefix/GPU_0/final_pred_heatmaps_tmp:0')
    cap = cv2.VideoCapture(0)
    centerx = 160
    centery = 120
    size = 24
    color_ = (0, 255, 0)

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        for step in range(128):

            # 使用txt
            # image_raw12_crop = np.loadtxt('/home/chen/Documents/Mobile_hand/experiments/trained/depart/models/mv2_hourglass_batch-64_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass_heatmap/test_real_image.txt', delimiter=',')
            # image_raw12_crop = image_raw12_crop[step, :]
            # image_raw12_crop = np.reshape(image_raw12_crop, [32, 32, 3])
            # tmp = image_raw12_crop[0, 1, :]
            # 使用cam
            ret, frame = cap.read()
            frame = scipy.misc.imresize(frame, (240, 320))

            image_raw1_crop = np.array(frame[centery - size: centery + size, centerx - size: centerx + size])
            image_raw1_crop = cv2.resize(image_raw1_crop, (int(32), int(32)), interpolation=cv2.INTER_AREA)

            image_raw1_crop = image_raw1_crop/255.0-0.5

            first_point = (centerx - size, centery - size)
            last_point = (centerx + size, centery + size)
            cv2.rectangle(frame, first_point, last_point, color=color_, thickness=2)

            image_raw1_crop = np.loadtxt('/home/chen/Documents/Mobile_hand/experiments/trained/depart/models/mv2_hourglass_batch-64_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass_heatmap/test_real_image.txt', delimiter=',')
            image_raw1_crop = image_raw1_crop[0,:]
            image_raw1_crop = np.reshape(image_raw1_crop, [32, 32, 3])

            preheat_v = sess.run(y, feed_dict={x: image_raw1_crop[np.newaxis, :]})
            preheat_v_tmp = preheat_v[0,:,:,0]
            # 根据preheat_v 计算最有可能的指尖坐标，当手指指尖存在时更新坐标centerx， centery
            if True:  # pre_is_loss_v[0, 0] > pre_is_loss_v[0, 1]:
                sum_preheat_v = np.sum(np.sum(np.sum(preheat_v, axis=0), axis=0), axis=0)
                print(sum_preheat_v)
                color_ = (0, 255, 0)
                motion = preheat_v[0, :, :, 0] - preheat_v[0, :, :, 1]
                raw, column = motion.shape
                _positon = np.argmax(motion)  # get the index of max in the a
                m, n = divmod(_positon, column)
            else:
                color_ = (255, 0, 0)
                m = 15.5
                n = 15.5

            # right_move = int((n - 15.5) / 32 * 48)
            # down_move = int((m - 15.5) / 32 * 48)
            # centery = centery + down_move
            # centerx = centerx + right_move
            input_image_v = (image_raw1_crop + 0.5) * 255
            input_image_v = input_image_v.astype(np.int16)

            if centery < 0 or centery > 240:
                centery = 120

            if centerx < 0 or centerx > 320:
                centerx = 160

            fig = plt.figure(1)
            plt.clf()
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(frame)  # 第一个batch的维度 hand1(0~31) back1(32~63)

            ax1 = fig.add_subplot(2, 2, 2)
            ax1.imshow(input_image_v)  # 第一个batch的维度 hand1(0~31) back1(32~63)

            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(preheat_v[0, :, :, 0])  # 第一个batch的维度 hand1(0~31) back1(32~63)
            ax3.set_title(np.sum(preheat_v[0, :, :, 0]))

            ax7 = fig.add_subplot(2, 2, 4)
            ax7.imshow(preheat_v[0, :, :, 1])  # 第一个batch的维度 hand1(0~31) back1(32~63)

            if np.sum(preheat_v[0, :, :, 0])>5:
                print(np.sum(preheat_v[0, :, :, 0]))
                image_raw1_crop = np.concatenate([image_raw1_crop[np.newaxis, :],image_raw1_crop[np.newaxis, :]],axis=0)
                image_raw1_crop = np.reshape(image_raw1_crop, [2,-1])
                np.savetxt("/home/chen/Documents/Mobile_hand/experiments/trained/depart/models/mv2_hourglass_batch-64_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass_heatmap/test_real_image.txt",
                           image_raw1_crop, fmt='%f', delimiter=',')



            plt.savefig(
                "/home/chen/Documents/Mobile_hand/experiments/varify/image/valid_on_cam/softmax/" + str(step).zfill(
                    10) + "_.png")
            plt.pause(0.01)