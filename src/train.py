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
from mpl_toolkits.mplot3d import Axes3D
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def get_loss_and_output(model, batchsize, input_image, input_heat, reuse_variables=None):
    losses = []

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        _, pred_heatmaps_all = get_network(model, input_image, True)

    for idx, pred_heat in enumerate(pred_heatmaps_all):
        loss_l2 = tf.nn.l2_loss(tf.concat(pred_heat, axis=0) - input_heat, name='loss_heatmap_stage%d' % idx)
        losses.append(loss_l2)

    total_loss = tf.reduce_sum(losses) / batchsize
    total_loss_ll_heat = tf.reduce_sum(loss_l2) / batchsize
    return total_loss, total_loss_ll_heat, pred_heat


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
        dataset_RHD = RHD()

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
                    input_image = batch_data_all[0]
                    input_heat = batch_data_all[3]

                    loss, last_heat_loss, pred_heat = get_loss_and_output(params['model'], params['batchsize'],
                                                                          input_image, input_heat, reuse_variable)
                    reuse_variable = True
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

        saver = tf.train.Saver(max_to_keep=5)

        tf.summary.scalar("learning_rate", learning_rate)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("loss_lastlayer_heat", last_heat_loss)

        summary_merge_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        # occupy gpu gracefully
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init.run()
            checkpoint_path = os.path.join(params['modelpath'], training_name)

            if checkpoint_path:
                saver.restore(sess, checkpoint_path+'/model-768500')
                print("restore from " + checkpoint_path)

            summary_writer = tf.summary.FileWriter(os.path.join(params['logpath'], training_name), sess.graph)
            total_step_num = params['num_train_samples'] * params['max_epoch'] // (params['batchsize'] * params['gpus'])
            print("Start training...")
            for step in tqdm(range(total_step_num)):
                _, loss_value, lh_loss = sess.run([train_op, loss, last_heat_loss])
                if step % params['per_update_tensorboard_step'] == 0:
                    """
                    summary_ = sess.run(summary_merge_op)
                    summary_writer.add_summary(summary_, step)
                    """
                    valid_loss_value, valid_lh_loss, valid_in_image, valid_in_heat, valid_p_heat = sess.run(
                        [loss, last_heat_loss, input_image, input_heat, pred_heat])

                    fig = plt.figure(1)
                    ax1 = fig.add_subplot('331')
                    ax2 = fig.add_subplot('332')
                    ax3 = fig.add_subplot('333')
                    ax2.set_title("loss:" + str(valid_loss_value))
                    ax4 = fig.add_subplot('334')
                    ax3.set_title("loss:" + str(valid_lh_loss))
                    ax5 = fig.add_subplot('335')
                    ax6 = fig.add_subplot('336')
                    ax7 = fig.add_subplot('337')
                    ax8 = fig.add_subplot('338')
                    ax9 = fig.add_subplot('339')

                    image = (valid_in_image + 0.5) * 255
                    image = image.astype(np.int16)
                    ax1.imshow(image[0,:,:,:])
                    valid_in_heat = valid_in_heat[0,:,:,:]
                    valid_p_heat = valid_p_heat[0,:,:,:]
                    valid_in_heat_a = np.sum(valid_in_heat,-1)
                    valid_p_heat_a = np.sum(valid_p_heat, -1)
                    ax2.imshow(valid_in_heat_a)
                    ax3.imshow(valid_p_heat_a)

                    ax4.imshow(valid_in_heat[:,:,0])
                    ax7.imshow(valid_p_heat[:, :, 0])
                    ax4.set_title("plam")
                    ax7.set_title("plam")

                    ax5.imshow(valid_in_heat[:,:,1])
                    ax8.imshow(valid_p_heat[:, :, 1])
                    ax5.set_title("thum")
                    ax8.set_title("thum")

                    ax6.imshow(valid_in_heat[:,:,5])
                    ax9.imshow(valid_p_heat[:, :, 5])
                    ax9.set_title("index")
                    ax6.set_title("index")



                    plt.savefig(os.path.join(params['logpath'], training_name)+"/no"+str(step).zfill(10)+".png")


                # save model
                if step % params['per_saved_model_step'] == 0:
                    saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=step)
                    print("loss_value: "+str(loss_value))

if __name__ == '__main__':
    tf.app.run()
