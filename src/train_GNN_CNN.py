#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys
import time
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from tqdm import tqdm
from utils.general import LearningRateScheduler, load_weights_from_snapshot

from RGB_db_interface.GANerate import GANerate
from dataset_interface.RHD import RHD
# training parameters



def log_line(logfile, msg):
    with open(logfile, 'a') as log_fh:
        log_fh.write(msg + '\n')
    print(msg)

train_para = {'lr': [1e-4, 1e-5, 1e-6],
              'lr_iter': [10000, 20000],
              'max_iter': 30000,
              'show_loss_freq': 1000,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_posenet'}

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# get dataset
dataset_GANerate = GANerate(batchnum=32)
image_crop, keypoint_uv21, keypoint_uv_heatmap, keypoint_xyz21_normed = dataset_GANerate.get_batch_data
image_crop_eval, keypoint_uv21_eval, keypoint_uv_heatmap_eval, keypoint_xyz21_normed_eval= dataset_GANerate.get_batch_data_eval


# build network
evaluation = tf.placeholder_with_default(True, shape=())
net = ColorHandPose3DNetwork()



keypoints_scoremap = net.inference_pose2d(image_crop, train=True)

with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    keypoints_scoremap_eval = net.inference_pose2d(image_crop_eval, train=True)
s =keypoint_uv_heatmap.get_shape().as_list()
keypoints_scoremap = [tf.image.resize_images(x, (s[1], s[2])) for x in keypoints_scoremap]
keypoints_scoremap_eval = [tf.image.resize_images(x, (s[1], s[2])) for x in keypoints_scoremap_eval]

# Loss
loss = 0.0
for i, pred_item in enumerate(keypoints_scoremap):
    loss += tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(pred_item - keypoint_uv_heatmap), [1, 2])))

# Loss
loss_eval = 0.0
for i, pred_item in enumerate(keypoints_scoremap_eval):
    loss_eval += tf.reduce_sum(tf.sqrt(tf.reduce_mean(tf.square(pred_item - keypoint_uv_heatmap_eval), [1, 2])))


# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=4.0)

rename_dict = {'CPM/PoseNet': 'PoseNet2D',
               '_CPM': ''}
load_weights_from_snapshot(sess, './weights/cpm-model-mpii', ['PersonNet', 'PoseNet/Mconv', 'conv5_2_CPM'], rename_dict)

# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
print('Starting to train ...')


early_stopping_metric = sys.float_info.max
waiting = 0
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
for epoch in range(1, int(140000/32/1000)*100 + 1):
    log_line(train_para['snapshot_dir']+'/train'+now+'.log', "== Epoch %i" % epoch)
    for one_epoch in tqdm(range(1000)):
        _, loss_v = sess.run([train_op, loss])

    print("\r\x1b[K", end='')
    log_line(train_para['snapshot_dir']+'/train'+now+'.log', " Train: loss: %.5f" % loss_v)
    loss_eval_v = 0.0
    for one_epoch in tqdm(range(100)):
        loss_eval_v += sess.run(loss_eval)
    loss_eval_v = loss_eval_v/100
    log_line(train_para['snapshot_dir'] + '/eval'+now+'.log', " Eval: loss: %.5f" % loss_eval_v)

    if early_stopping_metric > loss_eval_v:
        early_stopping_metric = loss_eval_v
        waiting = 0
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
        log_line(train_para['snapshot_dir'] + '/eval' + now + '.log', " save one model")
    else:
        waiting = waiting+1

    if waiting > 25:
        break

print('Training finished.')

