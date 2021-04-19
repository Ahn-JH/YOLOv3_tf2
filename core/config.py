from easydict import EasyDict as edict
import sys
sys.path.insert(0,'/home/jihun/TF2/')
from core.argument import call_args, cal_epoch
import argparse

args=call_args().parse_args()
__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the YOLO configuration
__C.YOLO.SEED                 = args.SEED
__C.YOLO.NAME                 = args.N
__C.YOLO.NETWORK              = args.Ar
__C.YOLO.ACTIVATION           = args.Activation
__C.YOLO.LOSS                 = args.Loss
__C.YOLO.NETWORK_WRITE        = True
__C.YOLO.CLASSES              = f"./data/classes/{args.C}"
__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = f"./data/dataset/{args.Train}"
__C.TRAIN.BATCH_SIZE          = 32
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = False
__C.TRAIN.LR_INIT             = float(args.IL)
__C.TRAIN.LR_END              = float(args.EL)
__C.TRAIN.WARMUP_EPOCHS       = 2  if args.WSTEP == 0 else cal_epoch(cfg,args.WSTEP)
__C.TRAIN.EPOCHS              = 20 if args.STEP  == 0 else cal_epoch(cfg,args.STEP)
print(__C.TRAIN.EPOCHS)

# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = f"./data/dataset/{args.Test}"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.SCORE_THRESHOLD      = 0.5
__C.TEST.IOU_THRESHOLD        = 0.4


