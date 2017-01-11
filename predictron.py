'''
A TensorFlow implementation of
The Predictron: End-To-End Learning and Planning
Silver et al.
https://arxiv.org/abs/1612.08810
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.losses as losses
import tensorflow.contrib.slim as slim

from util import predictron_arg_scope

logging.basicConfig()
logger = logging.getLogger('predictron')
logger.setLevel(logging.INFO)


class Predictron(object):
  def __init__(self, maze_ims, maze_labels, config):
    # self.inputs = tf.placeholder(tf.float32, shape=[None, config.maze_size, config.maze_size, 1])
    # self.targets = tf.placeholder(tf.float32, shape=[None, 20])

    self.inputs = maze_ims
    self.targets = maze_labels

    self.maze_size = config.maze_size
    self.max_depth = config.max_depth
    self.learning_rate = config.learning_rate
    self.max_grad_norm = config.max_grad_norm

    # Tensor rewards with shape [batch_size, max_depth + 1, maze_size]
    self.rewards = None
    # Tensor gammas with shape [batch_size, max_depth + 1, maze_size]
    self.gammas = None
    # Tensor lambdas with shape [batch_size, max_depth, maze_size]
    self.lambdas = None
    # Tensor values with shape [batch_size, max_depth + 1, maze_size]
    self.values = None
    # Tensor  preturns with shape [batch_size, max_depth + 1, maze_size]
    self.preturns = None
    # Tensor lambda_preturns with shape [batch_size, maze_size]
    self.lambda_preturns = None

    self.sess = tf.Session()
    self.graph = self.sess.graph

  def build(self):
    logger.info('Buidling Predictron.')
    self.build_model()
    self.build_loss()

    logger.info('Trainable variables:')
    logger.info('*' * 30)
    for var in tf.trainable_variables():
      logger.info(var.op.name)
    logger.info('*' * 30)

  def iter_func(self, state):
    sc = predictron_arg_scope()

    with slim.arg_scope(sc):
      net = slim.conv2d(state, 32, [3, 3], scope='conv1')
      net = layers.batch_norm(net, activation_fn=tf.nn.relu, scope='conv1/preact')

      with tf.variable_scope('value'):
        value_net = slim.fully_connected(slim.flatten(net), 32, scope='fc0')
        value_net = layers.batch_norm(value_net, activation_fn=tf.nn.relu, scope='fc0/preact')
        value_net = slim.fully_connected(value_net, self.maze_size, activation_fn=None, scope='fc1')

      net = slim.conv2d(net, 32, [3, 3], scope='conv2')
      net = layers.batch_norm(net, activation_fn=tf.nn.relu, scope='conv2/preact')
      net_flatten = slim.flatten(net, scope='conv2/flatten')

      with tf.variable_scope('reward'):
        reward_net = slim.fully_connected(net_flatten, 32, scope='fc0')
        reward_net = layers.batch_norm(reward_net, activation_fn=tf.nn.relu, scope='fc0/preact')
        reward_net = slim.fully_connected(reward_net, self.maze_size, activation_fn=None, scope='fc1')

      with tf.variable_scope('gamma'):
        gamma_net = slim.fully_connected(net_flatten, 32, scope='fc0')
        gamma_net = layers.batch_norm(gamma_net, activation_fn=tf.nn.relu, scope='fc0/preact')
        gamma_net = slim.fully_connected(gamma_net, self.maze_size, activation_fn=tf.nn.sigmoid, scope='fc1')

      with tf.variable_scope('lambda'):
        lambda_net = slim.fully_connected(net_flatten, 32, scope='fc0')
        lambda_net = layers.batch_norm(lambda_net, activation_fn=tf.nn.relu, scope='fc0/preact')
        lambda_net = slim.fully_connected(lambda_net, self.maze_size, activation_fn=tf.nn.sigmoid, scope='fc1')

      net = slim.conv2d(net, 32, [3, 3], scope='conv3')
      net = layers.batch_norm(net, activation_fn=tf.nn.relu, scope='conv3/preact')
    return net, reward_net, gamma_net, lambda_net, value_net

  def build_model(self):
    sc = predictron_arg_scope()
    with tf.variable_scope('state'):
      with slim.arg_scope(sc):
        state = slim.conv2d(self.inputs, 32, [3, 3], scope='conv1')
        state = layers.batch_norm(state, activation_fn=tf.nn.relu, scope='conv1/preact')
        state = slim.conv2d(state, 32, [3, 3], scope='conv2')
        state = layers.batch_norm(state, activation_fn=tf.nn.relu, scope='conv2/preact')

    iter_template = tf.make_template('iter', self.iter_func, unique_name_='iter')

    rewards_arr = []
    gammas_arr = []
    lambdas_arr = []
    values_arr = []

    for k in xrange(self.max_depth):
      state, reward, gamma, lambda_, value = iter_template(state)
      rewards_arr.append(reward)
      gammas_arr.append(gamma)
      lambdas_arr.append(lambda_)
      values_arr.append(value)

    _, _, _, _, value = iter_template(state)
    # K + 1 elements
    values_arr.append(value)

    bs = tf.shape(self.inputs)[0]
    # [batch_size, K * maze_size]
    self.rewards = tf.pack(rewards_arr, axis=1)
    # [batch_size, K, maze_size]
    self.rewards = tf.reshape(self.rewards, [bs, self.max_depth, self.maze_size])
    # [batch_size, K + 1, maze_size]
    self.rewards = tf.concat_v2(value=[tf.zeros(shape=[bs, 1, self.maze_size], dtype=tf.float32), self.rewards],
                                axis=1, name='rewards')

    # [batch_size, K * maze_size]
    self.gammas = tf.pack(gammas_arr, axis=1)
    # [batch_size, K, maze_size]
    self.gammas = tf.reshape(self.gammas, [bs, self.max_depth, self.maze_size])
    # [batch_size, K + 1, maze_size]
    self.gammas = tf.concat_v2(value=[tf.ones(shape=[bs, 1, self.maze_size], dtype=tf.float32), self.gammas],
                               axis=1, name='gammas')

    # [batch_size, K * maze_size]
    self.lambdas = tf.pack(lambdas_arr, axis=1)
    # [batch_size, K, maze_size]
    self.lambdas = tf.reshape(self.lambdas, [-1, self.max_depth, self.maze_size])

    # [batch_size, (K + 1) * maze_size]
    self.values = tf.pack(values_arr, axis=1)
    # [batch_size, K + 1, maze_size]
    self.values = tf.reshape(self.values, [-1, (self.max_depth + 1), self.maze_size])

    self.build_preturns()
    self.build_lambda_preturns()

  def build_preturns(self):
    ''' Eqn (2) '''

    g_preturns = []
    # for k = 0, g_0 = v[0], still fits.
    for k in xrange(self.max_depth, -1, -1):
      g_k = self.values[:, k, :]
      for kk in xrange(k, 0, -1):
        g_k = self.rewards[:, kk, :] + self.gammas[:, kk, :] * g_k
      g_preturns.append(g_k)
    # reverse to make 0...K from K...0
    g_preturns = g_preturns[::-1]
    self.g_preturns = tf.pack(g_preturns, axis=1, name='preturns')
    self.g_preturns = tf.reshape(self.g_preturns, [-1, self.max_depth + 1, self.maze_size])

  def build_lambda_preturns(self):
    ''' Eqn (4) '''
    g_k = self.values[:, -1, :]
    for k in xrange(self.max_depth - 1, -1, -1):
      g_k = (1 - self.lambdas[:, k, :]) * self.values[:, k, :] + \
            self.lambdas[:, k, :] * (self.rewards[:, k + 1, :] + self.gammas[:, k + 1, :] * g_k)
    self.g_lambda_preturns = g_k

  def build_loss(self):
    with tf.variable_scope('loss'):
      # Loss Eqn (5)
      # [batch_size, 1, maze_size]
      self.targets_tiled = tf.expand_dims(self.targets, 1)
      # [batch_size, K + 1, maze_size]
      self.targets_tiled = tf.tile(self.targets_tiled, [1, self.max_depth + 1, 1])
      self.loss_preturns = losses.mean_squared_error(self.g_preturns, self.targets_tiled, scope='preturns')
      losses.add_loss(self.loss_preturns)
      tf.summary.scalar('loss_preturns', self.loss_preturns)
      # Loss Eqn (7)
      self.loss_lambda_preturns = losses.mean_squared_error(
        self.g_lambda_preturns, self.targets, scope='lambda_preturns')
      losses.add_loss(self.loss_lambda_preturns)
      tf.summary.scalar('loss_lambda_preturns', self.loss_lambda_preturns)
      self.total_loss = losses.get_total_loss(name='total_loss')
