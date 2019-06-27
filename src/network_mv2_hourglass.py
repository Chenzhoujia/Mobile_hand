# -*- coding: utf-8 -*-
# @Time    : 18-4-12 5:12 PM
# @Author  : edvard_hua@live.com
# @FileName: network_mv2_cpm.py
# @Software: PyCharm

import tensorflow as tf
import tensorflow.contrib.slim as slim

from src.general import NetworkOps

ops = NetworkOps
from src.network_base import max_pool, upsample, inverted_bottleneck, separable_conv, convb, is_trainable

N_KPOINTS = 1
STAGE_NUM = 4

out_channel_ratio = lambda d: int(d * 1.0)
up_channel_ratio = lambda d: int(d * 1.0)



def hourglass_module(inp, stage_nums, l2s):
    if stage_nums > 0:
        down_sample = inp

        block_front = ops.conv_relu(down_sample, 'hourglass_front_0'+str(stage_nums), 3, 1, out_channel_ratio(24), True)
        block_front = ops.conv_relu(block_front, 'hourglass_front_1'+str(stage_nums), 3, 1, out_channel_ratio(24), True)
        block_front = ops.conv_relu(block_front, 'hourglass_front_2'+str(stage_nums), 3, 1, out_channel_ratio(24), True)
        block_front = ops.conv_relu(block_front, 'hourglass_front_3'+str(stage_nums), 3, 1, out_channel_ratio(24), True)
        block_front = ops.conv_relu(block_front, 'hourglass_front_4'+str(stage_nums), 3, 1, out_channel_ratio(24), True)


        stage_nums -= 1
        block_mid = hourglass_module(block_front, stage_nums, l2s)

        block_back = ops.conv_relu(block_mid, 'hourglass_back_' + str(stage_nums), 3, 1, N_KPOINTS, True)

        up_sample = block_back

        branch_jump = ops.conv_relu(inp, 'hourglass_branch_jump_0'+str(stage_nums), 3, 1, out_channel_ratio(24), True)
        branch_jump = ops.conv_relu(branch_jump, 'hourglass_branch_jump_1'+str(stage_nums), 3, 1, out_channel_ratio(24), True)
        branch_jump = ops.conv_relu(branch_jump, 'hourglass_branch_jump_2'+str(stage_nums), 3, 1, out_channel_ratio(24), True)
        branch_jump = ops.conv_relu(branch_jump, 'hourglass_branch_jump_3'+str(stage_nums), 3, 1, out_channel_ratio(24), True)
        branch_jump = ops.conv_relu(branch_jump, 'hourglass_branch_jump_4'+str(stage_nums), 3, 1, N_KPOINTS, True)

        curr_hg_out = tf.add(up_sample, branch_jump, name="hourglass_out_%d" % stage_nums)
        # mid supervise
        l2s.append(curr_hg_out)

        return curr_hg_out

    _ = ops.conv_relu(inp, 'hourglass_mid_' + str(stage_nums), 3, 1, out_channel_ratio(24), True)

    return _


def build_network(input, trainable):
    l2s = []
    is_trainable(trainable)

    net = ops.conv_relu(input, 'Conv2d_0', 3, 1, out_channel_ratio(16), True)

    net = ops.conv_relu(net, 'Conv2d_1', 3, 1, out_channel_ratio(16), True)
    net = ops.conv_relu(net, 'Conv2d_2', 3, 1, out_channel_ratio(16), True)

    net = ops.conv_relu(net, 'Conv2d_3', 3, 1, out_channel_ratio(24), True)
    net = ops.conv_relu(net, 'Conv2d_4', 3, 1, out_channel_ratio(24), True)
    net = ops.conv_relu(net, 'Conv2d_5', 3, 1, out_channel_ratio(24), True)
    net = ops.conv_relu(net, 'Conv2d_6', 3, 1, out_channel_ratio(24), True)
    net = ops.conv_relu(net, 'Conv2d_7', 3, 1, out_channel_ratio(24), True)

    net_h_w = int(net.shape[1])
    # build network recursively
    hg_out = hourglass_module(net, STAGE_NUM, l2s)

    for index, l2 in enumerate(l2s):
        l2_w_h = int(l2.shape[1])
        if l2_w_h == net_h_w:
            continue
        scale = net_h_w // l2_w_h
        l2s[index] = upsample(l2, scale, name="upsample_for_loss_%d" % index)

    # for index, l2 in enumerate(l2s):
    #     scale = 4
    #     l2s[index] = upsample(l2, scale, name="upsample2_for_loss_%d" % index)

    return hg_out, l2s
