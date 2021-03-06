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
sys.path.append('/home/jihun/TF2/')
import tensorflow as tf
import core.common as common
import core.utils as utils
import tensorflow.keras.layers as keras
from core.config import cfg


def Darknet(name=None):
    SIZE            = cfg.TRAIN.INPUT_SIZE
    inputs = keras.Input([SIZE, SIZE, 3])

    x = common.convolutional(inputs, filters_shape=(3, 3, 16), name='init_conv')
    x = common.convolutional(x,      filters_shape=(1, 1, 16), downsample=True, name='init_conv1')
    x = common.convolutional(x,      filters_shape=(3, 3, 32), downsample=True, name='DownSample1')

    x = common.Residual_block(x, 32, 1, name="Residual1")

    x = common.convolutional(x, filters_shape=(3, 3, 64), downsample=True, name='DownSample2')
    x = common.Residual_block(x, 64, 2, name="Residual2")

    route_2 = x

    x = common.convolutional(x, filters_shape=(3, 3, 128), downsample=True, name='DownSample3')
    x = common.Residual_block(x, 128, 4, name="Residual3")

    route_1 = x
    x = common.convolutional(x, filters_shape=(3, 3, 256), downsample=True, name='DownSample4')
    x = common.convolutional(x, filters_shape=(3, 3, 512), name='Last_conv')
    route_0 = common.Residual_block(x, 512, 2, name="Residual5")

    return tf.keras.Model(inputs, (route_2, route_1, route_0), name='Darknet')

def YOLOv3(sizes):
    NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
    SIZE            = cfg.TRAIN.INPUT_SIZE
    route_2  = keras.Input(sizes[0][1:], dtype=tf.float16)
    route_1  = keras.Input(sizes[1][1:], dtype=tf.float16)
    route_0  = keras.Input(sizes[2][1:], dtype=tf.float16)

   
    conv = common.convolutional(route_0, (1, 1, 256), name='Lconv1')
    conv = common.convolutional(conv,    (3, 3, 512), name='Lconv2')
    conv = common.convolutional(conv,    (1, 1, 256), name='Lconv3')
    conv = common.convolutional(conv,    (3, 3, 512), name='Lconv4')
    conv = common.convolutional(conv,    (1, 1, 256), name='Lconv5')

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512), name='Lbranch')
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3*(NUM_CLASS + 5)), activate=False, bn=False, name='Ldetect')
    conv_lbbox = tf.cast(conv_lbbox, tf.float32)

    conv = common.convolutional(conv, (1, 1, 128), name='LMconnect')
    conv = common.upsample(conv, name='Lupsample')

    conv = tf.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 128), name='Mconv1')
    conv = common.convolutional(conv, (3, 3, 256), name='Mconv2')
    conv = common.convolutional(conv, (1, 1, 128), name='Mconv3')
    conv = common.convolutional(conv, (3, 3, 256), name='Mconv4')
    conv = common.convolutional(conv, (1, 1, 128), name='Mconv5')

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256), name='Mbranch')
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3*(NUM_CLASS + 5)), activate=False, bn=False, name='Mdetect')
    conv_mbbox = tf.cast(conv_mbbox, tf.float32)

    conv = common.convolutional(conv, (1, 1, 64), name='MSconnect')
    conv = common.upsample(conv, name='Mupsample')

    conv = tf.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1,  64), name='Sconv1')
    conv = common.convolutional(conv, (3, 3, 128), name='Sconv2')
    conv = common.convolutional(conv, (1, 1,  64), name='Sconv3')
    conv = common.convolutional(conv, (3, 3, 128), name='Sconv4')
    conv = common.convolutional(conv, (1, 1,  64), name='Sconv5')

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128), name='Sbranch')
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 128, 3*(NUM_CLASS +5)), activate=False, bn=False, name='Sdetect')
    conv_sbbox = tf.cast(conv_sbbox, tf.float32)

    return tf.keras.Model((route_2, route_1, route_0), (conv_sbbox, conv_mbbox, conv_lbbox), name='Detection')


