# Copyright 2018 Zihua Zeng (edvard_hua@live.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===================================================================================
# -*- coding: utf-8 -*-
#--checkpoint=/home/chen/Documents/Mobile_hand/experiments/trained/mv2_hourglass_deep/models/mv2_hourglass_batch-16_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass/model-227500
# --output_graph=/home/chen/Documents/Mobile_hand/experiments/trained/mv2_hourglass_deep/models/mv2_hourglass_batch-16_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass/model-227500.pb
# --size=32
# --model=mv2_hourglass
import tensorflow as tf
import argparse
import os

from pprint import pprint

from src import network_mv2_hourglass
from src.networks import get_network
from src.general import NetworkOps

ops = NetworkOps
checkpoint_path = '/home/chen/Documents/Mobile_hand/experiments/trained/depart/models/mv2_hourglass_batch-64_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass_heatmap/'
model_name = 'model-7000'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--model', type=str, default='mv2_hourglass', help='')
parser.add_argument('--size', type=int, default=32)
parser.add_argument('--checkpoint', type=str, default=checkpoint_path + model_name, help='checkpoint path')
parser.add_argument('--output_node_names', type=str, default='GPU_0/final_pred_heatmaps_tmp') #['GPU_0/final_r_Variable','GPU_0/final_x_Variable','GPU_0/final_y_Variable','GPU_0/final_z_Variable'])
parser.add_argument('--output_graph', type=str, default=checkpoint_path + model_name+'.pb', help='output_freeze_path')

args = parser.parse_args()
i = 0
batchsize = 1
def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor], name=name)

with tf.Graph().as_default(), tf.device("/cpu:0"):
    with tf.device("/gpu:%d" % i):
        with tf.name_scope("GPU_%d" % i):
            input_node = tf.placeholder(tf.float32, shape=[1, args.size, args.size, 3], name="input_image")
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                network_mv2_hourglass.N_KPOINTS = 2
                network_mv2_hourglass.STAGE_NUM = 2
                _, pred_heatmaps_all = get_network('mv2_hourglass', input_node, True)
            for loss_i in range(len(pred_heatmaps_all)):
                pred_heatmaps_tmp = pred_heatmaps_all[loss_i]
                pred_heatmaps_tmp = tf.nn.softmax(pred_heatmaps_tmp)

            output_node_ufxuz = tf.add(pred_heatmaps_tmp, 0, name='final_pred_heatmaps_tmp') #(1,4)
    saver = tf.train.Saver(max_to_keep=10)
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    # occupy gpu gracefully
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init.run()

        saver.restore(sess, args.checkpoint)
        print("restore from " + args.checkpoint)

        input_graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session
            input_graph_def,  # input_graph_def is useful for retrieving the nodes
            args.output_node_names.split(",")
        )

with tf.gfile.FastGFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())


"""
cd ~/Downloads/tensorflow 
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/home/chen/Documents/Mobile_hand/experiments/trained/depart/models/mv2_hourglass_batch-64_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass_heatmap/model-4200.pb

source activate TFlite
tflite_convert \
--graph_def_file=/home/chen/Documents/Mobile_hand/experiments/trained/depart/models/mv2_hourglass_batch-64_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass_heatmap/model-7000.pb \
--output_file=/home/chen/Documents/Mobile_hand/experiments/trained/depart/models/mv2_hourglass_batch-64_lr-0.001_gpus-1_32x32_..-experiments-mv2_hourglass_heatmap/model-7000.lite \
--output_format=TFLITE \
--input_shapes=1,32,32,3 \
--input_arrays=GPU_0/input_image \
--output_arrays=GPU_0/final_pred_heatmaps_tmp \
--inference_type=FLOAT
"""
