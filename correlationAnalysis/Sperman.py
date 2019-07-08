import matplotlib
import pandas as pd
import numpy as np
# Sperman秩相关系数
import scipy.io as sio
import _pickle as cPickle
import time, os, math
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    # >>> angle_between((1, 0, 0), (0, 1, 0))
    # 1.5707963267948966
    # >>> angle_between((1, 0, 0), (1, 0, 0))
    # 0.0
    # >>> angle_between((1, 0, 0), (-1, 0, 0))
    # 3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    angle = angle * 180 / math.pi
    return angle

from mpl_toolkits.mplot3d import Axes3D

src_dir = '/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/NYU/train/'
def loadAnnotation():
    '''is_trun:
        True: to load 14 joints from self.keep_list
        False: to load all joints
    '''
    path = os.path.join(src_dir, 'joint_data.mat')
    mat = sio.loadmat(path)
    camera_num = 1 #if self.subset == 'testing' else 3
    joint_xyz = [mat['joint_xyz'][idx] for idx in range(camera_num)]
    joint_uvd = [mat['joint_uvd'][idx] for idx in range(camera_num)]
    return joint_uvd

def visualize(uvd_pt, step):
    #观察序列，查看关键点坐标，确定角度由哪些坐标计算
    fig = plt.figure(1)
    fig.clear()
    ax2 = fig.add_subplot(121, projection='3d')

    ax2.scatter(uvd_pt[:, 0], uvd_pt[:, 1],uvd_pt[:, 2])
    ax2.view_init(azim=45.0, elev=20.0)  # aligns the 3d coord with the camera view
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_xlim((180, 280))
    ax2.set_ylim((160, 320))
    ax2.set_zlim((680, 900))

    ax = fig.add_subplot(122, projection='3d')
    ax.view_init(azim=45.0, elev=20.0)  # aligns the 3d coord with the camera view
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim((180, 280))
    ax.set_ylim((160, 320))
    ax.set_zlim((680, 900))
    fig_color = ['c', 'm', 'y', 'g', 'r', 'b']
    for f in range(6):
        if f < 5:
            for bone in range(5):
                ax.plot([uvd_pt[f * 6+bone, 0], uvd_pt[f * 6 +bone+ 1, 0]],
                        [uvd_pt[f * 6+bone, 1], uvd_pt[f * 6 +bone+ 1, 1]],
                        [uvd_pt[f * 6+bone, 2], uvd_pt[f * 6 +bone+ 1, 2]], color=fig_color[f], linewidth=2)
            if f == 4:
                ax.plot([uvd_pt[f * 6 + bone + 1, 0], uvd_pt[30, 0]],
                        [uvd_pt[f * 6 + bone + 1, 1], uvd_pt[30, 1]],
                        [uvd_pt[f * 6 + bone + 1, 2], uvd_pt[30, 2]], color=fig_color[f], linewidth=2)
            else:
                ax.plot([uvd_pt[f * 6 + bone + 1, 0], uvd_pt[34, 0]],
                        [uvd_pt[f * 6 + bone + 1, 1], uvd_pt[34, 1]],
                        [uvd_pt[f * 6 + bone + 1, 2], uvd_pt[34, 2]], color=fig_color[f], linewidth=2)
        # ax.scatter(uvd_pt[f * 2, 0], uvd_pt[f * 2, 1], uvd_pt[f * 2, 2], s=60, c=fig_color[f])
        ax.scatter(uvd_pt[f*6:(f+1)*6, 0], uvd_pt[f*6:(f+1)*6, 1], uvd_pt[f*6:(f+1)*6, 2], s=30, c=fig_color[f])

    plt.savefig("/tmp/image/" + str(step).zfill(5) + ".jpg")
    # plt.show()
    # plt.pause(0.01)

relation = np.loadtxt('relation.txt', delimiter=',')
no_relation = np.loadtxt('no_relation.txt', delimiter=',')
for k in range(15):
    relation[k, k] = 0.0


preframe = np.zeros((5, 3))
nowframe = np.zeros((5, 3))
diffframe = np.zeros((72756, 5, 3))

joints = loadAnnotation()
joints = joints[0]
for i in tqdm(range(72757)):
    j = i
    uvd_pt = joints[j]
    # 给nowframe赋值
    for f in range(5):
        # 012 uvd_pt[f*6] uvd_pt[f*6 + 1] uvd_pt[f*6 + 2]
        nowframe[f][0] = angle_between(uvd_pt[f*6] - uvd_pt[f*6 + 1], uvd_pt[f*6 + 2] - uvd_pt[f*6 + 1])
        # 234 uvd_pt[f*6 + 2] uvd_pt[f*6 + 3] uvd_pt[f*6 + 4]
        nowframe[f][1] = angle_between(uvd_pt[f*6 + 2] - uvd_pt[f*6 + 3], uvd_pt[f*6 + 4] - uvd_pt[f*6 + 3])
        if f == 4:
            # 45? uvd_pt[f*6 + 4] uvd_pt[f*6 + 5] uvd_pt[30]
            nowframe[f][2] = angle_between(uvd_pt[f*6 + 4] - uvd_pt[f*6 + 5], uvd_pt[30] - uvd_pt[f*6 + 5])
        else:
            # 45? uvd_pt[f*6 + 4] uvd_pt[f*6 + 5] uvd_pt[34]
            nowframe[f][2] = angle_between(uvd_pt[f*6 + 4] - uvd_pt[f*6 + 5], uvd_pt[34] - uvd_pt[f*6 + 5])
    if i > 0:
        diffframe[i-1] = nowframe-preframe
    #print(diffframe[i-1])
    preframe = np.array(nowframe)
    #visualize(uvd_pt, j)
diffframe = diffframe.reshape((72756, -1))
from scipy.stats import spearmanr

relation = np.zeros((15, 15))
no_relation = np.zeros((15, 15))
for x in tqdm(range(15)):
    for y in range(15):
        relation[x, y], no_relation[x, y] = spearmanr(diffframe[:, x], diffframe[:, y])

np.savetxt("relation.txt", relation, fmt='%f', delimiter=',')
np.savetxt("no_relation.txt", no_relation, fmt='%f', delimiter=',')