"""Utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers


def predictron_arg_scope(weight_decay=0.0001,
                         batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,
                         batch_norm_scale=True):
  batch_norm_params = {
    'decay': batch_norm_decay,
    'epsilon': batch_norm_epsilon,
    'scale': batch_norm_scale,
    'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  # Set weight_decay for weights in Conv and FC layers.
  with arg_scope(
      [layers.conv2d, layers_lib.fully_connected],
      weights_regularizer=regularizers.l2_regularizer(weight_decay)):
    with arg_scope(
        [layers.conv2d],
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=None,
        normalizer_fn=layers_lib.batch_norm,
        normalizer_params=batch_norm_params) as sc:
      return sc


class Colour():
  '''
  Borrowed from https://github.com/brendanator/predictron/blob/master/predictron/util.py
  Copyright (c) 2016 Brendan Maginnis
  MIT Licence
  '''
  NORMAL = '\033[0m'

  BLACK = '\033[30m'
  RED = '\033[31m'
  GREEN = '\033[32m'
  YELLOW = '\033[33m'
  BLUE = '\033[34m'
  MAGENTA = '\033[35m'
  CYAN = '\033[36m'
  WHITE = '\033[37m'

  @classmethod
  def highlight(cls, input_, success):
    if success:
      colour = Colour.GREEN
    else:
      colour = Colour.RED
    return colour + input_ + Colour.NORMAL
