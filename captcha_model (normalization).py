#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 23:00:57 2017

@author: dell
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import captcha_input
import config

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT
CLASSES_NUM = config.CLASSES_NUM
CHARS_NUM = config.CHARS_NUM

rate = 0.01

def inputs(train, batch_size):
    return captcha_input.inputs(train, batch_size=batch_size)

def _conv(name, input, size, input_channels, output_channels, is_training=True):
    with tf.variable_scope(name) as scope:
        if not is_training:
            scope.reuse_variables()
        kernel = _weight_variable('weights', shape=[size, size ,input_channels, output_channels])
        biases = _bias_variable('biases',[output_channels])
        pre_activation = tf.nn.bias_add(_conv2d(input, kernel),biases)
        conv = tf.maximum(rate*pre_activation,pre_activation, name=scope.name)
        return conv

def _conv2d(value, weight):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(value, weight, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(value, name, is_training):
  """max_pool_2x2 downsamples a feature map by 2X."""
  with tf.variable_scope(name) as scope1:
    if not is_training:
      scope1.reuse_variables()
    return tf.nn.max_pool(value, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)


def _weight_variable(name, shape):
  """weight_variable generates a weight variable of a given shape."""
  initializer = tf.truncated_normal_initializer(stddev=0.1)
  var = tf.get_variable(name,shape,initializer=initializer, dtype=tf.float32)
  return var


def _bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  initializer = tf.constant_initializer(0.1)
  var = tf.get_variable(name, shape, initializer=initializer,dtype=tf.float32)
  return var

def _batch_norm(name,inputs, is_training):
  """ Batch Normalization
  """
  with tf.variable_scope(name):
#      return tf.layers.batch_normalization(input,training=is_training)
    return tf.contrib.layers.batch_norm(inputs,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)
def inference(images, keep_prob, is_training=True):
  images = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  # 128,448

  conv1 = _conv('conv1', images, 3, 1, 64, is_training)
    
  pool1 = _max_pool_2x2(conv1, name='pool1', is_training=is_training)
  
  conv2 = _conv('conv2', pool1, 3, 64, 64, is_training)
    
  pool2 = _max_pool_2x2(conv2, name='pool2', is_training=is_training)
  
  conv3 = _conv('conv3', pool2, 3, 64, 128, is_training)
  conv4 = _conv('conv4', conv3, 1, 128, 64, is_training)
  conv5 = _conv('conv5', conv4, 3, 64, 128, is_training)
    
  pool3 = _max_pool_2x2(conv5, name='pool3', is_training=is_training)
  
  conv6 = _conv('conv6', pool3, 3, 128, 256, is_training)
  conv7 = _conv('conv7', conv6, 1, 256, 128, is_training)
  conv8 = _conv('conv8', conv7, 3, 128, 256, is_training)
    
  pool4 = _max_pool_2x2(conv8, name='pool4', is_training=is_training) # 8,28
  
  conv9 = _conv('conv9', pool4, 3, 256, 256, is_training)
  pool5 = _max_pool_2x2(conv9, name='pool5', is_training=is_training) # 4,14
  conv10 = _conv('conv10', pool5, 3, 256, 256, is_training)
  conv11 = _conv('conv11', conv10, 3, 256, 128, is_training)
  pool6 = _max_pool_2x2(conv11, name='pool6', is_training=is_training) # 2,7,128
  conv_norm = _batch_norm('norm', pool6, is_training)
  
  
  with tf.variable_scope('local1') as scope:
    if not is_training:
      scope.reuse_variables()
    batch_size = images.get_shape()[0].value
    reshape = tf.reshape(conv_norm, [batch_size,-1])
    dim = reshape.get_shape()[1].value
    weights = _weight_variable('weights', shape=[dim,1024])
    biases = _bias_variable('biases',[1024])
    local1 = tf.nn.relu(tf.matmul(reshape,weights) + biases, name=scope.name)

    local1_drop = tf.nn.dropout(local1, keep_prob)

  with tf.variable_scope('softmax_linear') as scope:
    if not is_training:
      scope.reuse_variables()
    weights = _weight_variable('weights',shape=[1024,CHARS_NUM*CLASSES_NUM])
    biases = _bias_variable('biases',[CHARS_NUM*CLASSES_NUM])
    softmax_linear = tf.add(tf.matmul(local1_drop,weights), biases, name=scope.name)

  return tf.reshape(softmax_linear, [-1, CHARS_NUM, CLASSES_NUM])


def loss(logits, labels):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                  labels=labels, logits=logits, name='corss_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def training(loss):
  optimizer = tf.train.AdamOptimizer(1e-4)
  train_op = optimizer.minimize(loss)
  return train_op

#def training(loss):
#  optimizer = tf.train.AdamOptimizer(1e-4, 0.5)
#  
#  gen_grads_and_vars = optimizer.compute_gradients(loss)
#  gen_train = optimizer.apply_gradients(gen_grads_and_vars)
#  
#  ema = tf.train.ExponentialMovingAverage(decay=0.99)
#  update_losses = ema.apply([loss])
#
#  global_step = tf.contrib.framework.get_or_create_global_step()
#  incr_global_step = tf.assign(global_step, global_step+1)
#
#  return tf.group(update_losses, incr_global_step, gen_train)
  
  

def evaluation(logits, labels):
  correct_prediction = tf.equal(tf.argmax(logits,2), tf.argmax(labels,2))
  correct_batch = tf.reduce_mean(tf.cast(correct_prediction, tf.int32), 1)
  return tf.reduce_sum(tf.cast(correct_batch, tf.float32))


def output(logits):
  return tf.argmax(logits, 2)

