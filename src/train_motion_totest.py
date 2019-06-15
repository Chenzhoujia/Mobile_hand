# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import platform
import time
import numpy as np
import configparser

from tqdm import tqdm

from dataset_interface.RHD import RHD
from dataset_interface.dataset_prepare import CocoPose
from src.networks import get_network
import matplotlib.pyplot as plt
from src.general import NetworkOps
from src import  network_mv2_hourglass
from mpl_toolkits.mplot3d import Axes3D
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
ops = NetworkOps
def upsample(inputs, factor, name):
    return tf.image.resize_bilinear(inputs, [int(inputs.get_shape()[1]) * factor, int(inputs.get_shape()[2]) * factor],
                                    name=name)
def get_loss_and_output(model, batchsize, scoremap, hand_motion, reuse_variables=None):

    with tf.variable_scope("diff", reuse=reuse_variables):
        network_mv2_hourglass.N_KPOINTS = 1
        _, pred_diffmap_all = get_network(model, scoremap, True)
    losses = []
    for idx, pred_heat in enumerate(pred_diffmap_all):
        # flatten
        s = pred_heat.get_shape().as_list()
        pred_heat = tf.reshape(pred_heat, [-1, s[3]*s[1]*s[2]])  # this is Bx16*16*1

        # pred_heat --> 3 params
        out_chan_list = [32, 16, 8]
        for i, out_chan in enumerate(out_chan_list):
            pred_heat = ops.fully_connected_relu(pred_heat, 'fc_vp_%d_%d' %(idx,i), out_chan=out_chan, trainable=True)
            evaluation = tf.placeholder_with_default(True, shape=())
            pred_heat = pred_heat# ops.dropout(pred_heat, 0.95, evaluation)

        ux = ops.fully_connected(pred_heat, 'fc_vp_ux_%d' % idx, out_chan=1, trainable=True)
        uy = ops.fully_connected(pred_heat, 'fc_vp_uy_%d' % idx, out_chan=1, trainable=True)
        uz = ops.fully_connected(pred_heat, 'fc_vp_uz_%d' % idx, out_chan=1, trainable=True)
        ur = ops.fully_connected(pred_heat, 'fc_vp_ur_%d' % idx, out_chan=1, trainable=True)

        loss_l2r = tf.nn.l2_loss(hand_motion[:, 0] - ur[:, 0], name='lossr_heatmap_stage%d' % idx)
        loss_l2x = tf.nn.l2_loss(hand_motion[:, 1] - ux[:, 0], name='lossx_heatmap_stage%d' % idx)
        loss_l2y = tf.nn.l2_loss(hand_motion[:, 2] - uy[:, 0], name='lossy_heatmap_stage%d' % idx)
        loss_l2z = tf.nn.l2_loss(hand_motion[:, 3] - uz[:, 0], name='lossz_heatmap_stage%d' % idx)
        losses.append(loss_l2x+loss_l2y+loss_l2r*0.001+loss_l2z*0.001)

    ufxuz = tf.concat(values=[ur, ux, uy, uz], axis=1, name='fxuz')

    motion_loss = tf.reduce_sum(losses) / batchsize
    alph = 0.5
    total_loss =motion_loss
    return total_loss, ufxuz


def average_gradients(tower_grads):
    """
    Get gradients of all variables.
    :param tower_grads:
    :return:
    """
    average_grads = []

    # get variable and gradients in differents gpus
    for grad_and_vars in zip(*tower_grads):
        # calculate the average gradient of each gpu
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def main(argv=None):
    # load config file and setup
    params = {}
    config = configparser.ConfigParser()
    config_file = "../experiments/mv2_cpm.cfg"
    if len(argv) != 1:
        config_file = argv[1]
    config.read(config_file)
    for _ in config.options("Train"):
        params[_] = eval(config.get("Train", _))

    os.environ['CUDA_VISIBLE_DEVICES'] = params['visible_devices']

    gpus_index = params['visible_devices'].split(",")
    params['gpus'] = len(gpus_index)

    if not os.path.exists(params['modelpath']):
        os.makedirs(params['modelpath'])
    if not os.path.exists(params['logpath']):
        os.makedirs(params['logpath'])

    gpus = 'gpus'
    if platform.system() == 'Darwin':
        gpus = 'cpu'
    training_name = '{}_batch-{}_lr-{}_{}-{}_{}x{}_{}'.format(
        params['model'],
        params['batchsize'],
        params['lr'],
        gpus,
        params['gpus'],
        params['input_width'], params['input_height'],
        config_file.replace("/", "-").replace(".cfg", "")
    )

    with tf.Graph().as_default(), tf.device("/cpu:0"):
        dataset_RHD = RHD(batchnum=params['batchsize'])

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(float(params['lr']), global_step,
                                                   decay_steps=10000, decay_rate=float(params['decay_rate']),
                                                   staircase=True)
        opt = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        tower_grads = []
        reuse_variable = False

        for i in range(params['gpus']):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope("GPU_%d" % i):
                    #input_image, keypoint_xyz, keypoint_uv, input_heat, keypoint_vis, k, num_px_left_hand, num_px_right_hand \
                    batch_data_all = dataset_RHD.get_batch_data

                    hand_motion = batch_data_all[9]
                    scoremap1 = batch_data_all[11]
                    scoremap2 = batch_data_all[12]
                    one_scoremap = tf.ones_like(scoremap1)
                    scoremap1 = tf.where(scoremap1 > 1, x=one_scoremap, y=scoremap1)
                    scoremap2 = tf.where(scoremap2 > 1, x=one_scoremap, y=scoremap2)

                    scoremap = scoremap1 - scoremap2

                    # 计算一个scoremap的loss
                    scoremap = tf.reduce_sum(scoremap, axis=-1)
                    scoremap = tf.expand_dims(scoremap, axis=-1)  # hand back

                    """
                    total_loss, motion_loss*0.00001, loss_scoremap*0.001, loss_is_loss,\
                               ur, ux, uy, uz, ufxuz, pred_heatmaps_tmp, pre_is_loss, is_loss12
                    """
                    loss, ufxuz\
                        = get_loss_and_output(params['model'], params['batchsize'],
                                              scoremap, hand_motion, reuse_variable)

                    loss_all = loss
                    grads = opt.compute_gradients(loss_all)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram("gradients_on_average/%s" % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        MOVING_AVERAGE_DECAY = 0.99
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variable_to_average)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(max_to_keep=10)

        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("loss", loss)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            checkpoint_path = os.path.join(params['modelpath'], training_name)
            model_name = '/model-65000'
            if checkpoint_path:
                saver.restore(sess, checkpoint_path+model_name)
                print("restore from " + checkpoint_path+model_name)

            summary_writer = tf.summary.FileWriter(os.path.join(params['logpath'], training_name), sess.graph)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize'] * params['gpus'])
            print("Start training...")
            for step in tqdm(range(total_step_num)):
                _, loss_value = sess.run([train_op, loss])
                if step % params['per_update_tensorboard_step'] == 0:
                    """
                                        loss, ufxuz\
                        = get_loss_and_output(params['model'], params['batchsize'],
                                              scoremap, hand_motion, reuse_variable)
                    """
                    loss_v, ufxuz_v, scoremap_v, hand_motion_v= sess.run(
                        [loss, ufxuz, scoremap, hand_motion])

                    fig = plt.figure(1)
                    plt.clf()
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(scoremap_v[0, :, :, 0])#第一个batch的维度 hand1(0~31) back1(32~63)
                    """
                            loss_l2r = tf.nn.l2_loss(hand_motion[:, 0] - ur[:, 0], name='lossr_heatmap_stage%d' % idx)
                            loss_l2x = tf.nn.l2_loss(hand_motion[:, 1] - ux[:, 0], name='lossx_heatmap_stage%d' % idx)
                            loss_l2y = tf.nn.l2_loss(hand_motion[:, 2] - uy[:, 0], name='lossy_heatmap_stage%d' % idx)
                            loss_l2z = tf.nn.l2_loss(hand_motion[:, 3] - uz[:, 0], name='lossz_heatmap_stage%d' % idx)
                            losses.append(loss_l2x+loss_l2y+loss_l2r*0.001+loss_l2z*0.001)
                    
                        ufxuz = tf.concat(values=[ur, ux, uy, uz], axis=1, name='fxuz')
                    """
                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.plot([0, hand_motion_v[0, 1]], [0, hand_motion_v[0, 2]], label= "label", color='red')
                    ax2.plot([0, ufxuz_v[0, 1]], [0, ufxuz_v[0, 2]], label="predict", color='blue')
                    ax2.set_xlim((-1, 1))
                    ax2.set_ylim((1, -1))
                    ax2.grid(True)

                    plt.savefig(os.path.join(params['logpath']) + "/" + str(step).zfill(10) + "_.png")

                    print("loss:"+str(loss_value))

                    # save model
                if step % params['per_saved_model_step'] == 0:
                    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=step)

if __name__ == '__main__':
    tf.app.run()
