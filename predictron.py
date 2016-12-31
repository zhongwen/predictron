'''
A TensorFlow implementation of
The Predictron: End-To-End Learning and Planning
Silver et al.
https://arxiv.org/abs/1612.08810
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.losses as losses


class Predictron(object):
  def __init__(self, config):
    self.inputs = tf.placeholder(tf.float32, shape=[None, config.maze_size, config.maze_size, 1])
    self.targets = tf.placeholder(tf.float32, shape=[None, 20])

    self.maze_size = config.maze_size
    self.max_depth = config.max_depth
    self.learning_rate = config.learning_rate
    self.max_grad_norm = config.max_grad_norm
    self.max_ckpts_to_keep = config.max_ckpts_to_keep

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

  def build(self):
    self.build_model()
    self.build_loss()
    self.setup_global_step()
    self.setup_train_op()
    self.setup_init_op()

  def iter_func(self, state):
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
        weights_regularizer=slim.l2_regularizer(0.0005)):
      with tf.variable_scope('hidden'):
        net = slim.conv2d(state, 32, [3, 3])

      with tf.variable_scope('val'):
        value_net = slim.fully_connected(slim.flatten(net), 32)
        value_net = slim.fully_connected(value_net, self.maze_size, activation_fn=None)

      net = slim.conv2d(net, 32, [3, 3])
      net_flatten = slim.flatten(net)
      with tf.variable_scope('reward'):
        reward_net = slim.fully_connected(net_flatten, 32)
        reward_net = slim.fully_connected(reward_net, self.maze_size, activation_fn=None)

      with tf.variable_scope('gamma'):
        gamma_net = slim.fully_connected(net_flatten, 32)
        gamma_net = slim.fully_connected(gamma_net, self.maze_size, activation_fn=tf.nn.sigmoid)

      with tf.variable_scope('lambda'):
        lambda_net = slim.fully_connected(net_flatten, 32)
        lambda_net = slim.fully_connected(lambda_net, self.maze_size, activation_fn=tf.nn.sigmoid)

      net = slim.conv2d(net, 32, [3, 3])
    return net, reward_net, gamma_net, lambda_net, value_net

  def build_model(self):
    with tf.variable_scope('state'):
      state = slim.conv2d(self.inputs, 32, [3,3], scope='conv1')
      state = slim.conv2d(state, 32, [3,3], scope='conv2')

    iter_template = tf.make_template('iter', self.iter_func)

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
    self.rewards = tf.concat(1, [tf.zeros(shape=[bs, 1, self.maze_size], dtype=tf.float32), self.rewards], 'rewards')

    # [batch_size, K * maze_size]
    self.gammas = tf.pack(gammas_arr, axis=1)
    # [batch_size, K, maze_size]
    self.gammas = tf.reshape(self.gammas, [bs, self.max_depth, self.maze_size])
    # [batch_size, K + 1, maze_size]
    self.gammas = tf.concat(1, [tf.ones(shape=[bs, 1, self.maze_size], dtype=tf.float32), self.gammas], 'gammas')

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


    self.saver = tf.train.Saver(max_to_keep=self.max_ckpts_to_keep)

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
            self.lambdas[:,k, :] * (self.rewards[:,k + 1,:] + self.gammas[:, k + 1,:] * g_k)
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
      # Loss Eqn (7)
      self.loss_lambda_preturns = losses.mean_squared_error(
        self.g_lambda_preturns, self.targets, scope='lambda_preturns')
      losses.add_loss(self.loss_lambda_preturns)
      self.total_loss = losses.get_total_loss(name='total_loss')



  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def setup_init_op(self):
    self.init_op = tf.global_variables_initializer()

  def setup_train_op(self):
    self.train_op = tf.contrib.layers.optimize_loss(
      loss=self.total_loss,
      global_step=self.global_step,
      learning_rate=self.learning_rate,
      optimizer=tf.train.AdamOptimizer,
      clip_gradients=self.max_grad_norm,
      name='train_op'
    )

  def train(self, maze_im, maze_target):
    self.sess.run(self.train_op,
                  feed_dict={self.inputs: maze_im,
                             self.targets: maze_target})

  def init(self):
    self.sess.run(self.init_op)
