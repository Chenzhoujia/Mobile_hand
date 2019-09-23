import tensorflow as tf
import argparse
import cv2
import numpy as np
from numpy import unravel_index
from tqdm import tqdm
import scipy.misc
import matplotlib.pyplot as plt
from RGB_db_interface.GANerate import plot_hand

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

    x = graph.get_tensor_by_name('prefix/input_node_representations:0')
    y = graph.get_tensor_by_name('prefix/final_output_node_representations:0')

    # We launch a Session
    import os
    import shutil
    shutil.rmtree('./snapshots_posenet/baseline/real_eval/')  # 能删除该文件夹和文件夹下所有文件
    os.mkdir('./snapshots_posenet/baseline/real_eval/')
    cap = cv2.VideoCapture(0)
    with tf.Session(graph=graph) as sess:

        for step in tqdm(range(1,1024)):
            """
            使用cv2.imread()接口读图像，读进来的是BGR格式以及【0～255】。所以只要将img转换为RGB格式显示即可：

            """
            #frame = cv2.imread('/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/GANeratedHands_Release/data/noObject/0001/'
            #                   + str(step).zfill(4) + '_color_composed.png')
            #frame = cv2.imread('/home/chen/Documents/Mobile_hand/src/snapshots_posenet/baseline/real/' + str(step) + '.jpg')
            #frame = cv2.imread('/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/ICCV2017/RHD_published_v2/evaluation/color/' + str(step) + '.jpg')

            ret, frame = cap.read()
            frame = frame[240 - 128:240 + 128, 320 - 128:320 + 128, :]

            frame = frame[:,:, [2, 1, 0]]
            frame = frame.astype(np.float)
            frame = frame / 255.0 - 0.5

            heatmap = sess.run(y, feed_dict={x: frame[np.newaxis, :]})

            keypoint_uv21_pre =  np.zeros([1,21,2])
            for i in range(heatmap.shape[0]):
                for j in range(heatmap.shape[-1]):
                    heatmap_pre_tmp = heatmap[i,:,:,j]
                    cor_tmp = unravel_index(heatmap_pre_tmp.argmax(), heatmap_pre_tmp.shape)
                    keypoint_uv21_pre[i,j,0] = cor_tmp[1]
                    keypoint_uv21_pre[i,j,1] = cor_tmp[0]



            frame = (frame + 0.5) * 255
            frame = frame.astype(np.int16)
            fig = plt.figure(1)
            plt.clf()
            ax1 = fig.add_subplot(121)
            ax1.imshow(frame)
            plot_hand(keypoint_uv21_pre[0], ax1)

            ax2 = fig.add_subplot(122)
            ax2.imshow(np.sum(heatmap[0], axis=-1))  # 第一个batch的维度 hand1(0~31) back1(32~63)
            ax2.scatter(keypoint_uv21_pre[0, :, 0], keypoint_uv21_pre[0, :, 1], s=10, c='k', marker='.')
            plt.pause(0.01)
            #plt.savefig('./snapshots_posenet/baseline/real_eval/' + str(step).zfill(5) + '.png')
