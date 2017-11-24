import tensorflow as tf

from tensorflow.contrib.layers.python.layers import initializers
from functools import reduce 

def conv2d(x_arg,
           output_dim,
           kernel_size,
           stride,
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer,
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d',
           trainable=True):
  with tf.variable_scope(name):
    if data_format == 'NCHW':
      stride = [1, 1, stride[0], stride[1]]
      kernel_shape = [kernel_size[0], kernel_size[1], x_arg.get_shape()[1], output_dim]
    elif data_format == 'NHWC':
      stride = [1, stride[0], stride[1], 1]
      kernel_shape = [kernel_size[0], kernel_size[1], x_arg.get_shape()[-1], output_dim]

    w_matrix = tf.get_variable('w', kernel_shape,
        tf.float32, initializer=weights_initializer, trainable=trainable)
    conv = tf.nn.conv2d(x_arg, w_matrix, stride, padding, data_format=data_format)

    b_matrix = tf.get_variable('b', [output_dim],
        tf.float32, initializer=biases_initializer, trainable=trainable)
    out = tf.nn.bias_add(conv, b_matrix, data_format)

  if activation_fn != None:
    out = activation_fn(out)

  return out, w_matrix, b_matrix

def linear(input_,
           output_size,
           weights_initializer=initializers.xavier_initializer(),
           biases_initializer=tf.zeros_initializer,
           activation_fn=None,
           trainable=True,
           name='linear'):
  shape = input_.get_shape().as_list()

  if len(shape) > 2:
    input_ = tf.reshape(input_, [-1, reduce(lambda x, y: x * y, shape[1:])])
    shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    w_matrix = tf.get_variable('w', [shape[1], output_size], tf.float32,
        initializer=weights_initializer, trainable=trainable)
    b_matrix = tf.get_variable('b', [output_size],
        initializer=biases_initializer, trainable=trainable)
    out = tf.nn.bias_add(tf.matmul(input_, w_matrix), b_matrix)

    if activation_fn != None:
      return activation_fn(out), w_matrix, b_matrix
    else:
      return out, w_matrix, b_matrix

def batch_sample(probs, name='batch_sample'):
  with tf.variable_scope(name):
    uniform = tf.random_uniform(tf.shape(probs), minval=0, maxval=1)
    samples = tf.argmax(probs - uniform, dimension=1)
  return samples
