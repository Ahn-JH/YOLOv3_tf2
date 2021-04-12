#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-07-11 23:37:51
#   Description :
#
#================================================================

import tensorflow as tf
import core.common as common

def Darknet(inputs,name=None):

    x = common.convolutional(inputs, filters_shape=(3, 3, 16), name='init_conv')
    x = common.convolutional(inputs, filters_shape=(3, 3, 32), name='init_conv1', downsample=True)
    x = common.convolutional(inputs, filters_shape=(3, 3, 64), name='init_conv2', downsample=True)

    x = common.Residual_block(x, 64, 1, name="Residual1")

    x = common.convolutional(x, filters_shape=(3, 3, 128), downsample=True, name='DownSample5')
    x = common.Residual_block(x, 128, 2, name="Residual2")

    route_2 = x

    x = common.convolutional(x, filters_shape=(3, 3, 256), downsample=True, name='DownSample6')
    x = common.Residual_block(x, 256, 4, name="Residual3")

    route_1 = x
    x = common.convolutional(x, filters_shape=(3, 3, 512), downsample=True, name='DownSample7')
    route_0 = common.Residual_block(x, 512, 2, name="Residual5")
    
    return route_2, route_1, route_0
    #return tf.keras.Model(inputs, (route_0, route_1, route_2), name=name)

#  def darknet53(input_data):
#
#      input_data = common.convolutional(input_data, (3, 3,  3,  32))
#      input_data = common.convolutional(input_data, (3, 3, 32,  64), downsample=True)
#
#      for i in range(1):
#          input_data = common.residual_block(input_data,  64,  32, 64)
#
#      input_data = common.convolutional(input_data, (3, 3,  64, 128), downsample=True)
#
#      for i in range(2):
#          input_data = common.residual_block(input_data, 128,  64, 128)
#
#      input_data = common.convolutional(input_data, (3, 3, 128, 256), downsample=True)
#
#      for i in range(8):
#          input_data = common.residual_block(input_data, 256, 128, 256)
#
#      route_1 = input_data
#      input_data = common.convolutional(input_data, (3, 3, 256, 512), downsample=True)
#
#      for i in range(8):
#          input_data = common.residual_block(input_data, 512, 256, 512)
#
#      route_2 = input_data
#      input_data = common.convolutional(input_data, (3, 3, 512, 1024), downsample=True)
#
#      for i in range(4):
#          input_data = common.residual_block(input_data, 1024, 512, 1024)
#
#      return route_1, route_2, input_data
#
#
