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
def get_loss_and_output(model, batchsize, input_image1,input_image2, hand_motion,scoremap1,scoremap2, reuse_variables=None):
    losses = []

    # 叠加在batch上重用特征提取网络
    input_image12 = tf.concat([input_image1, input_image2], 0)
    input_image12.set_shape([batchsize*2, 32, 32, 3])
    input_image12 = tf.add(input_image12, 0, name='input_image')
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        network_mv2_hourglass.N_KPOINTS = 1
        _, pred_heatmaps_all12 = get_network(model, input_image12, True)
    # 计算一个scoremap的loss
    scoremap12 = tf.concat([scoremap1, scoremap2], 0)
    scoremap12 = tf.reduce_sum(scoremap12,axis=-1)
    scoremap12 = tf.expand_dims(scoremap12,axis=-1)
    scoremap12.set_shape([batchsize * 2, 32, 32, 1])
    loss_scoremap = 0
    for loss_i in range(len(pred_heatmaps_all12)):
        scale = 2
        tmp = upsample(pred_heatmaps_all12[loss_i], scale, name="upsample_for_hotmap_loss_%d" % loss_i)
        #loss_scoremap += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tmp, labels=scoremap12))
        loss_scoremap += tf.nn.l2_loss(tmp-scoremap12)
    loss_scoremap=loss_scoremap/32.0/32.0/32.0
    diffmap = []
    for batch_i in range(len(pred_heatmaps_all12)):
        diffmap.append(pred_heatmaps_all12[batch_i][0:batchsize]-pred_heatmaps_all12[batch_i][batchsize:batchsize*2])

    #diffmap_t 将4个阶段的输出，在通道数上整合
    for batch_i in range(len(diffmap)):
        if batch_i==0:
            diffmap_t = diffmap[batch_i]
        else:
            diffmap_t = tf.concat([diffmap[batch_i], diffmap_t], axis=3)

    with tf.variable_scope("diff", reuse=reuse_variables):
        network_mv2_hourglass.N_KPOINTS = 1
        _, pred_diffmap_all = get_network(model, diffmap_t, True)
    losses = []
    for idx, pred_heat in enumerate(pred_diffmap_all):
        # flatten
        s = pred_heat.get_shape().as_list()
        pred_heat = tf.reshape(pred_heat, [s[0], -1])  # this is Bx16*16*1
        #x = tf.concat([x, hand_side], 1)

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
        losses.append(loss_l2x+loss_l2y+loss_l2r*0.1+loss_l2z*0.01)
    ufxuz = tf.concat(values=[ur, ux, uy, uz], axis=1, name='fxuz')

    total_loss = tf.reduce_sum(losses) / batchsize
    alph = 0.5
    total_loss = alph*loss_scoremap + (1-alph)*total_loss
    return total_loss, loss_scoremap, ur, ux, uy, uz, ufxuz, tmp


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
                    input_image1 = batch_data_all[8]
                    input_image2 = batch_data_all[10]
                    hand_motion = batch_data_all[9]
                    scoremap1 = batch_data_all[11]
                    scoremap2 = batch_data_all[12]

                    loss, loss_scoremap, ur, ux, uy, uz, ufxuz, preheat = get_loss_and_output(params['model'], params['batchsize'],
                                                                input_image1, input_image2, hand_motion, scoremap1,scoremap2,reuse_variable)
                    grads = opt.compute_gradients(loss)
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

        summary_merge_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            checkpoint_path = os.path.join(params['modelpath'], training_name)
            model_name = '/model-173000'
            if checkpoint_path:
                saver.restore(sess, checkpoint_path+model_name)
                print("restore from " + checkpoint_path+model_name)

            summary_writer = tf.summary.FileWriter(os.path.join(params['logpath'], training_name), sess.graph)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize'] * params['gpus'])
            print("Start training...")
            for step in tqdm(range(total_step_num)):
                _, loss_value = sess.run([train_op, loss])
                if step % params['per_saved_model_step'] == 0:
                    """
                    summary_ = sess.run(summary_merge_op)
                    summary_writer.add_summary(summary_, step)
                    """
                    valid_loss_value, valid_scoremap_loss, valid_input_image1, valid_input_image2, valid_hand_motion, \
                    ur_v, ux_v, uy_v, uz_v, preheat_v, scoremap1_v, scoremap2_v= sess.run(
                        [loss, loss_scoremap, input_image1, input_image2, hand_motion,
                         ur, ux, uy, uz, preheat, scoremap1, scoremap2])

                    valid_input_image1 = (valid_input_image1 + 0.5) * 255
                    valid_input_image1 = valid_input_image1.astype(np.int16)

                    valid_input_image2 = (valid_input_image2 + 0.5) * 255
                    valid_input_image2 = valid_input_image2.astype(np.int16)

                    fig = plt.figure(1)
                    plt.clf()
                    ax1 = fig.add_subplot('331')
                    ax1.imshow(valid_input_image1[0, :, :, :])
                    ax2 = fig.add_subplot('332')
                    ax2.imshow(valid_input_image2[0, :, :, :])
                    ax3 = fig.add_subplot('333')
                    ax3.plot([0, valid_hand_motion[0, 1]], [0, valid_hand_motion[0, 2]], label= "label", color='red')
                    ax3.plot([0, ux_v[0]], [0, uy_v[0]], label="predict", color='blue')
                    ax3.set_title("loss:" + str(valid_loss_value-valid_scoremap_loss*0.5))


                    ax4 = fig.add_subplot('334')
                    ax4.imshow(np.sum(scoremap1_v[0], axis=-1))
                    ax5 = fig.add_subplot('335')
                    ax5.imshow(np.sum(scoremap2_v[0], axis=-1))

                    ax7 = fig.add_subplot('337')
                    ax7.imshow(preheat_v[0, :, :, 0])
                    ax8 = fig.add_subplot('338')
                    ax8.imshow(preheat_v[32, :, :, 0])
                    ax7.set_title("loss:" + str(valid_scoremap_loss*0.5))
                    ax3.set_xlim((-1, 1))
                    ax3.set_ylim((-1, 1))
                    ax3.grid(True)
                    print(valid_loss_value)
                    print(valid_scoremap_loss*0.5)

                    plt.savefig(os.path.join(params['logpath'], training_name)+"/"+str(step).zfill(10)+".png")
                # save model
                if step % params['per_saved_model_step'] == 0:
                    #print("loss_value: "+str(loss_value))
                    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=step)
                    # used_vars = set()
                    # for variable in sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                    #     used_vars.add(variable.name)

if __name__ == '__main__':
    tf.app.run()
