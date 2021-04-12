#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
#================================================================
import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from importlib import import_module
from core.dataset import Dataset
from core.yolov3 import decode, compute_loss
from core.config import cfg
from core.utils import write_network
from core.network import YOLOv3, Darknet
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from load_checkpoint import load_checkpoint
from core.argument import call_args, cal_epoch
import argparse

args=call_args().parse_args()

# Graphic Card Setting
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

#Using Tensor core for rtx Graphic Card 
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

#  tf.config.experimental_run_functions_eagerly(True)

#Set result Dir
root_path      = os.getcwd()
proj_dir       = f'{root_path}/Projs/{cfg.YOLO.NAME}'
checkpoint_dir = f'{root_path}/Projs/{cfg.YOLO.NAME}/checkpoint'
logdir         = f'{root_path}/Projs/{cfg.YOLO.NAME}/log'
mAPdir         = f'{root_path}/Projs/{cfg.YOLO.NAME}/mAP'

if args.Load == False:
    if os.path.exists(proj_dir)       : shutil.rmtree(proj_dir)
    if os.path.exists(checkpoint_dir) : shutil.rmtree(checkpoint_dir)
    if os.path.exists(logdir)         : shutil.rmtree(logdir)
    if os.path.exists(mAPdir)         : shutil.rmtree(mAPdir)

    os.mkdir(proj_dir)
    os.mkdir(checkpoint_dir)
    os.mkdir(logdir)
    os.mkdir(mAPdir)

print(cfg.YOLO.SEED)
tf.random.set_seed(int(cfg.YOLO.SEED))
trainset = Dataset('train')

STEPS_PER_EPOCH = len(trainset)
GLOBAL_STEPS    = tf.Variable(1, trainable=False, dtype=tf.int64)
WARMUP_STEPS    = cfg.TRAIN.WARMUP_EPOCHS * STEPS_PER_EPOCH
TOTAL_STEPS     = cfg.TRAIN.EPOCHS * STEPS_PER_EPOCH

network         = import_module(f'core.Architecture.{cfg.YOLO.NETWORK}.network')
YOLOv3, Darknet = network.YOLOv3, network.Darknet
input_tensor    = tf.keras.layers.Input([416, 416, 3])

def build(input_tensor):
    route_2, route_1, route_0 = Darknet()(input_tensor)

    tensor_sizes = (route_2.get_shape(),
                    route_1.get_shape(),
                    route_0.get_shape())

    conv_tensors   = YOLOv3(tensor_sizes)((route_2,route_1,route_0))
    output_tensors = []

    for i, conv_tensor in enumerate(conv_tensors):
        conv_shape   = conv_tensor.get_shape()[1:]
        pred_tensor  = decode(conv_shape, i)(conv_tensor)
        output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    return output_tensors


optimizer   = tf.keras.optimizers.Adam()
model       = tf.keras.Model(input_tensor, build(input_tensor), name='YOLO')
if args.Load: model.load_weights(load_checkpoint(checkpoint_dir))
writer      = tf.summary.create_file_writer(logdir)

if cfg.YOLO.NETWORK_WRITE: write_network(model, proj_dir)

print(model.summary())

@tf.function
def train_step(image_data, target):

    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]
        
        total_loss = giou_loss + conf_loss + prob_loss
        gradients  = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # update learning rate
        GLOBAL_STEPS.assign_add(1)
        if GLOBAL_STEPS < WARMUP_STEPS:
            lr = GLOBAL_STEPS / WARMUP_STEPS * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((GLOBAL_STEPS - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS) * np.pi)))
        lr=tf.cast(lr, tf.float32, name='LR')
        optimizer.lr.assign(lr)

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr",              optimizer.lr,  step=GLOBAL_STEPS)
            tf.summary.scalar("loss/total_loss", total_loss  ,  step=GLOBAL_STEPS)
            tf.summary.scalar("loss/giou_loss" , giou_loss   ,  step=GLOBAL_STEPS)
            tf.summary.scalar("loss/conf_loss" , conf_loss   ,  step=GLOBAL_STEPS)
            tf.summary.scalar("loss/prob_loss" , prob_loss   ,  step=GLOBAL_STEPS)
        writer.flush()

        return [GLOBAL_STEPS,TOTAL_STEPS, optimizer.lr, giou_loss, conf_loss, prob_loss, total_loss]

START = time.time()
for epoch in range(cfg.TRAIN.EPOCHS):
    EPOCH_TIME = time.time()
    count = 0
    for image_data, target in trainset:
        info=train_step(image_data, target)
        print("=> STEP %4d / %4d lr: %.6f | giou_loss: %4.2f | conf_loss: %4.2f | prob_loss: %4.2f | total_loss: %4.2f"
                %(info[0],info[1],info[2],info[3],info[4],info[5],info[6]))
        count += 1
    END = time.time()
    Accumulated_time = round(END - START,1)
    Epoch_time =round((END - EPOCH_TIME)/count,2)
    Done = round(Epoch_time * count * (cfg.TRAIN.EPOCHS - epoch ),1)
    print(f"=> EPOCH  {epoch+1}/{cfg.TRAIN.EPOCHS} | TIME : {Accumulated_time}s | SPEED/STEP : {Epoch_time}s | Simulation will done in {Done} Seconds")

    if epoch > cfg.TRAIN.EPOCHS - 5:
        model.save_weights(f"{checkpoint_dir}/yolov3_{epoch}")

with open(proj_dir + '/Structure.txt' , 'a') as f:
    f.write(f"\n TIME : {-START+END}\n")



