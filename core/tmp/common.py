#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : common.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:12:53
#   Description :
#
#================================================================

import tensorflow as tf

class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """
    def call(self, x, training=False, name=None):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape,downsample=False, activate=True, bn=True, name=None):
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    bn_name = f'{name}_BN' if name != None else None
    ac_name = f'{name}_ACTIVATE' if name != None else None

    conv = tf.keras.layers.Conv2D(filters=filters_shape[2], kernel_size = filters_shape[0:2], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.), name=name)(input_layer)



    if bn: conv = BatchNormalization()(conv,name=bn_name)
    if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1,name=ac_name)
    return conv

def residual_network(input_layer, filters, i, Rname=None):
    prev = input_layer
    name = f'{Rname}/conv{i*2+1}/' if Rname != None else None
    conv = convolutional(input_layer, filters_shape=(1, 1, filters // 2) , name=name)
    name = f'{Rname}/conv{i*2+2}/' if Rname != None else None
    conv = convolutional(conv       , filters_shape=(3, 3, filters), name=name)
    residual_ouput = tf.keras.layers.Add()([prev,conv])
    return residual_ouput

def Residual_block(input_layer, filters, blocks, name=None):

    for i in range(blocks):
        Residual_layer = residual_network(input_layer, filters, i, name)

    return Residual_layer

def upsample(input_layer, name=None):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest',name=None)

