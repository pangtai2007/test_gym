import os
import tensorflow as tf

from .layers import *
from .network import Network

class CNN(Network):
  def __init__(self, sess,
               data_format,
               history_length,
               observation_dims,
               output_size,
               trainable=True,
               hidden_activation_fn=tf.nn.relu,
               output_activation_fn=None,
               weights_initializer=initializers.xavier_initializer(),
               biases_initializer=tf.constant_initializer(0.1),
               value_hidden_sizes=[512],
               advantage_hidden_sizes=[512],
               network_output_type='dueling',
               network_header_type='nips',
               name='CNN'):
    super(CNN, self).__init__(sess, name)

    if data_format == 'NHWC':
      self.inputs = tf.placeholder('float32',
          [None] + observation_dims + [history_length], name='inputs')
    elif data_format == 'NCHW':
      self.inputs = tf.placeholder('float32',
          [None, history_length] + observation_dims, name='inputs')
    else:
      raise ValueError("unknown data_format : %s" % data_format)

    self.var = {}
    self.layer0 = tf.div(self.inputs, 255.)

    with tf.variable_scope(name):
      if network_header_type.lower() == 'nature':
        self.layer4, self.var['l4_w'], self.var['l4_b'] = \
            linear(self.layer0, 512, weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l4_conv')
        layer = self.layer4
      elif network_header_type.lower() == 'nips':
        self.layer3, self.var['l3_w'], self.var['l3_b'] = \
            linear(self.layer0, 256, weights_initializer, biases_initializer,
            hidden_activation_fn, data_format, name='l3_conv')
        layer = self.layer3
      else:
        raise ValueError('Wrong DQN type: %s' % network_header_type)

      self.build_output_ops(layer, network_output_type,
          value_hidden_sizes, advantage_hidden_sizes, output_size,
          weights_initializer, biases_initializer, hidden_activation_fn,
          output_activation_fn, trainable)
