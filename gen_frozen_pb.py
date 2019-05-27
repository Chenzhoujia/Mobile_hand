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
#--checkpoint=/home/chen/Documents/Mobile_hand/experiments/trained/mv2_hourglass_deep/models/mv2_hourglass_batch-16_lr-0.001_gpus-1_192x192_..-experiments-mv2_hourglass/model-5500 --output_graph=/home/chen/Documents/Mobile_hand/experiments/trained/mv2_hourglass_deep/models/mv2_hourglass_batch-16_lr-0.001_gpus-1_192x192_..-experiments-mv2_hourglass/model-5500.pb --size=192 --model=mv2_hourglass
import tensorflow as tf
import argparse
import os

from pprint import pprint

from src.networks import get_network

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
parser.add_argument('--model', type=str, default='mv2_cpm', help='')
parser.add_argument('--size', type=int, default=224)
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path')
parser.add_argument('--output_node_names', type=str, default='upsample2_for_loss_3')
parser.add_argument('--output_graph', type=str, default='./model.pb', help='output_freeze_path')

args = parser.parse_args()

input_node = tf.placeholder(tf.float32, shape=[1, args.size, args.size, 3], name="image")

with tf.Session() as sess:
    net = get_network(args.model, input_node, trainable=False)
    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)

    input_graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,  # The session
        input_graph_def,  # input_graph_def is useful for retrieving the nodes
        args.output_node_names.split(",")
    )

with tf.gfile.GFile(args.output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
