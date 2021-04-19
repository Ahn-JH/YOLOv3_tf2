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

import sys
import tensorflow as tf
import core.common as common
import core.utils as utils
import tensorflow.keras.layers as keras
from core.config import cfg
from importlib import import_module
common = import_module(f'core.common_{cfg.YOLO.Activation}')



def Darknet(name=None):
    SIZE            = cfg.TRAIN.INPUT_SIZE
    inputs = keras.Input([SIZE, SIZE, 3])

    x = common.convolutional(inputs, filters_shape=(3, 3, 12), name='init_conv')
    x = common.convolutional(x,      filters_shape=(3, 3, 24), name='conv1', downsample=True)

    x = common.Residual_block(x, 24, 1, name="Residual1")
    x = common.convolutional(x,      filters_shape=(3, 3, 48), name='conv2', downsample=True)

    x = common.Residual_block(x, 48, 1, name="Residual2")
    x = common.convolutional(x,      filters_shape=(3, 3, 96), name='conv3', downsample=True)
    x = common.Residual_block(x, 96, 4, name="Residual3")

    route_2 = x

    x = common.convolutional(x, filters_shape=(3, 3, 192), name='conv4', downsample=True)
    x = common.Residual_block(x, 192, 4, name="Residual4")

    route_1 = x
    x = common.convolutional(x, filters_shape=(3, 3, 384), downsample=True, name='conv5')
    route_0 = common.Residual_block(x, 384, 2, name="Residual5")

    return tf.keras.Model(inputs, (route_2, route_1, route_0), name='Darknet')

def YOLOv3(sizes):
    NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
    SIZE            = cfg.TRAIN.INPUT_SIZE
    route_2  = keras.Input(sizes[0][1:], dtype=tf.float16)
    route_1  = keras.Input(sizes[1][1:], dtype=tf.float16)
    route_0  = keras.Input(sizes[2][1:], dtype=tf.float16)

   
    conv = common.convolutional(route_0, (1, 1, 192), name='Lconv1')
    conv = common.convolutional(conv,    (3, 3, 384), name='Lconv2')
    conv = common.convolutional(conv,    (1, 1, 192), name='Lconv3')
    conv = common.convolutional(conv,    (3, 3, 384), name='Lconv4')
    conv = common.convolutional(conv,    (1, 1, 192), name='Lconv5')

    conv_lobj_branch = common.convolutional(conv, (3, 3, 384), name='Lbranch')
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 384, 3*(NUM_CLASS + 5)), activate=False, bn=False, name='Ldetect')
    conv_lbbox = tf.cast(conv_lbbox, tf.float32)

    conv = common.convolutional(conv, (1, 1, 96), name='LMconnect')
    conv = common.upsample(conv, name='Lupsample')

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 96 ), name='Mconv1')
    conv = common.convolutional(conv, (3, 3, 192), name='Mconv2')
    conv = common.convolutional(conv, (1, 1, 96 ), name='Mconv3')
    conv = common.convolutional(conv, (3, 3, 192), name='Mconv4')
    conv = common.convolutional(conv, (1, 1, 96 ), name='Mconv5')

    conv_mobj_branch = common.convolutional(conv, (3, 3, 192), name='Mbranch')
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 192, 3*(NUM_CLASS + 5)), activate=False, bn=False, name='Mdetect')
    conv_mbbox = tf.cast(conv_mbbox, tf.float32)

    conv = common.convolutional(conv, (1, 1, 48), name='MSconnect')
    conv = common.upsample(conv, name='Mupsample')

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 48), name='Sconv1')
    conv = common.convolutional(conv, (3, 3, 96), name='Sconv2')
    conv = common.convolutional(conv, (1, 1, 48), name='Sconv3')
    conv = common.convolutional(conv, (3, 3, 96), name='Sconv4')
    conv = common.convolutional(conv, (1, 1, 48), name='Sconv5')

    conv_sobj_branch = common.convolutional(conv, (3, 3, 96), name='Sbranch')
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 96, 3*(NUM_CLASS +5)), activate=False, bn=False, name='Sdetect')
    conv_sbbox = tf.cast(conv_sbbox, tf.float32)

    return tf.keras.Model((route_2, route_1, route_0), (conv_sbbox, conv_mbbox, conv_lbbox), name='Detection')


