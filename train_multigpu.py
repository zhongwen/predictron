'''
Modified from Tensorflow/models/tutorials/image/cifar10/cifar10_multi_gpu_train.py
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import time
import os

import numpy as np
import tensorflow as tf

from predictron import Predictron
from maze import MazeGenerator

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './ckpts/predictron_train',
                           'dir to save checkpoints and TB logs')
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'num of batches')
tf.app.flags.DEFINE_integer('num_gpus', 2, 'num of GPUs to use')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')

tf.flags.DEFINE_integer('batch_size', 100, 'batch size')
tf.flags.DEFINE_integer('maze_size', 20, 'size of maze (square)')
tf.flags.DEFINE_float('maze_density', 0.3, 'Maze density')
tf.flags.DEFINE_integer('max_depth', 16, 'maximum model depth')
tf.flags.DEFINE_float('max_grad_norm', 10., 'clip grad norm into this value')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

def tower_loss(scope, maze_ims, maze_labels, config):
  model = Predictron(maze_ims, maze_labels, config)
  model.build()
  loss_preturns = model.loss_preturns
  loss_lambda_preturns = model.loss_lambda_preturns
  losses = tf.get_collection('losses', scope)
  total_loss = tf.add_n(losses, name='total_loss')
  return total_loss,loss_preturns, loss_lambda_preturns


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat_v2(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train():
  config = FLAGS
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
      'global_step', [],
      initializer=tf.constant_initializer(0), trainable=False)

    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    maze_ims_ph = tf.placeholder(tf.float32, shape=[None, FLAGS.maze_size, FLAGS.maze_size, 1])
    maze_labels_ph = tf.placeholder(tf.float32, shape=[None, FLAGS.maze_size])

    maze_ims_splits = tf.split(0, FLAGS.num_gpus, maze_ims_ph)
    maze_labels_splits = tf.split(0, FLAGS.num_gpus, maze_labels_ph)
    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % ('predictron', i)) as scope:
          # Calculate the loss for one tower of the CIFAR model. This function
          # constructs the entire CIFAR model but shares the variables across
          # all towers.
          loss, loss_preturns, loss_lambda_preturns = tower_loss(
            scope,
            maze_ims_splits[i],
            maze_labels_splits[i],
            config)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='predictron_0')
    update_op = tf.group(*update_ops)
    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, update_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    # TODO

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    maze_gen = MazeGenerator(
      height=FLAGS.maze_size,
      width=FLAGS.maze_size,
      density=FLAGS.maze_density)

    for step in xrange(FLAGS.max_steps):
      # TODO(zhongwen): make a seperate thread
      maze_ims, maze_labels = maze_gen.generate_labelled_mazes(FLAGS.batch_size)
      start_time = time.time()
      _, loss_value, loss_preturns_val, loss_lambda_preturns_val = sess.run(
        [train_op, loss, loss_preturns, loss_lambda_preturns],
        feed_dict={
          maze_ims_ph: maze_ims,
          maze_labels_ph: maze_labels
          })
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.4f, loss_preturns = %.4f, loss_lambda_preturns = %.4f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.datetime.now(), step, loss_value, loss_preturns_val, loss_lambda_preturns_val,
                             examples_per_sec, sec_per_batch))

      # if step % 100 == 0:
        # summary_str = sess.run(summary_op)
        # summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
  train()

if __name__ == '__main__':
  tf.app.run()