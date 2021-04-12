#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-07-19 10:29:34
#   Description :
#
#================================================================

import cv2
import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import decode
from load_checkpoint import load_checkpoint
from importlib import import_module
from tensorflow.keras.mixed_precision import experimental as mixed_precision
sys.path.insert(0,'/home/jihun/TF2/')
from core.argument import call_args, cal_epoch
import argparse

args=call_args().parse_args()


network = import_module(f'core.Architecture.{cfg.YOLO.NETWORK}.network')
YOLOv3, Darknet = network.YOLOv3, network.Darknet


tf.get_logger().setLevel('ERROR')
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
tf.config.experimental_run_functions_eagerly(False)
#Using Tensor core in rtx Graphic Card 
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)




INPUT_SIZE   = 416
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)
proj         = cfg.YOLO.NAME

root_dir                = os.getcwd() 
weight_dir              = f'{root_dir}/Projs/{proj}/checkpoint'
predicted_dir_path      = f'{root_dir}/Projs/{proj}/mAP/predicted'
ground_truth_dir_path   = f'{root_dir}/Projs/{proj}/mAP/ground-truth'
detected_image_dir_path = f'{root_dir}/Projs/{proj}/detection'


if os.path.exists(predicted_dir_path)     : shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path)  : shutil.rmtree(ground_truth_dir_path)
if os.path.exists(detected_image_dir_path): shutil.rmtree(detected_image_dir_path)

os.mkdir(predicted_dir_path)
os.mkdir(ground_truth_dir_path)
os.mkdir(detected_image_dir_path)

# Build Model
input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])

route_2, route_1, route_0 = Darknet()(input_layer)
sizes=(route_2.get_shape(),route_1.get_shape(),route_0.get_shape())
feature_maps = YOLOv3(sizes)((route_2,route_1,route_0))


bbox_tensors = []
for i, fm in enumerate(feature_maps):
    fm_shape = fm.get_shape()[1:]
    bbox_tensor = decode(fm_shape, i)(fm)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)

model.load_weights(load_checkpoint(weight_dir))

with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
    for num, line in enumerate(annotation_file):
        annotation = line.strip().split()
        image_path = annotation[0]
        image_name = image_path.split('/')[-1]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

        if len(bbox_data_gt) == 0:
            bboxes_gt=[]
            classes_gt=[]
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

        print('=> ground truth of %s:' % image_name)
        num_bbox_gt = len(bboxes_gt)
        with open(ground_truth_path, 'w') as f:
            for i in range(num_bbox_gt):
                class_name = CLASSES[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())
        print('=> predict result of %s:' % image_name)
        predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
        # Predict Process
        image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = model.predict(image_data)
        #  print(model.get_layer("Darknet").summary())
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')


        if detected_image_dir_path is not None:
            image = utils.draw_bbox(image, bboxes)
            cv2.imwrite(detected_image_dir_path+'/'+image_name, image)

        with open(predict_result_path, 'w') as f:
            for bbox in bboxes:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = CLASSES[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(bbox_mess)
                print('\t' + str(bbox_mess).strip())

