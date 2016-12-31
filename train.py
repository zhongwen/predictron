'''
Training part
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from predictron import Predictron
from maze import Maze

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('maze_size', 20, 'size of maze (square)')
tf.flags.DEFINE_integer('max_depth', 16, 'maximum model depth')
tf.flags.DEFINE_integer('num_steps', 1000000, 'num of training steps')
tf.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.flags.DEFINE_float('max_grad_norm', 10., 'clip grad norm into this value')
tf.flags.DEFINE_integer('max_ckpts_to_keep', 20, 'maximum checkpoint models to keep')
tf.flags.DEFINE_string('train_dir', '/tmp', 'training directory')
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")


def main(unused_argv):

  config = FLAGS

  model = Predictron(config)
  model.build()
  model.init()

  maze_gen = Maze(16, 20)
  for step in xrange(FLAGS.num_steps):
    maze_im, maze_label = maze_gen.get_batch()
    model.train(maze_im, maze_label)


if __name__ == '__main__':
  tf.app.run()
