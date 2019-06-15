import pickle
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
mask = scipy.misc.imread(os.path.join('/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/ICCV2017/RHD_published_v2/training/mask/', '%.5d.png' % 0))
"""
Segmentation masks available:
0: background, 1: person, 
2-4: left thumb [tip to palm], 5-7: left index, ..., 14-16: left pinky, 17: palm, 
18-20: right thumb, ..., right palm: 33
"""
mask =np.zeros_like(mask)
fig = plt.figure(1)
ax1 = fig.add_subplot('111')
ax1.imshow(mask)
plt.show()

im = Image.fromarray(mask)
im.save("zero.png")