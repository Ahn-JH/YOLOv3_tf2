import argparse
import numpy as np

def call_args():
    parser= argparse.ArgumentParser(description=
            'config file for YOLO version 3 in tensorflow |Required Option : Name, Train, Test')
    parser.add_argument('--Option', required=False, help='|R : Required Option')
    parser.add_argument('--N','--Name', required=False, default='Debug', type=str,help="All results will saved in Proj/[NAME] |R")
    parser.add_argument('--C','--Classes', required=False, default='class_NODI.names', type=str, help='Define class')
    parser.add_argument('--Freeze', required=False, type=str, help='Freeze weight, Option : [Darknet , Detection, Both]')
    parser.add_argument('--Train', required=False, default='defect_train.txt', type=str, help='Define Train set name | Default path : ./data/dataset/ |R')
    parser.add_argument('--Test', required=False, default='defect_test.txt', type=str, help='Define Test set name | Default path : ./data/dataset/ |R')
    parser.add_argument('--Ar','--Architecture', required=False, default='N0', type=str, help='Change the YOLO network, Architecture will follow the Proj name')
    parser.add_argument('--Ep','--Epoch', required=False, type=int, default=20, help='1 Epoch : len(dataset)/Batchsize Steps')
    parser.add_argument('--Warm', required=False, type=int, default=2, help='During Warmup Epoch, Learning Rate increase from 0 to init_Learning Rate')
    parser.add_argument('--IL','--init_LR', required=False,  default=1e-3, help='init Learning Rate , Defaults = 1e-3')
    parser.add_argument('--EL','--end_LR', required=False,  default=1e-8, help='end Learning Rate , Defaults = 1e-8')
    parser.add_argument('--SEED','--S', required=False, default=1, help='Set SEED value, defaults = 1')   
    parser.add_argument('--STEP', required=False, default=0,  type=int, help='Set total STEP, DO NOT USE WITH EPOCH OPTION')
    parser.add_argument('--WSTEP', required=False, default=0, type=int, help='Set Warmup STEP, DO NOT USE WITH Warm OPTION')
    parser.add_argument('--CHECKPOINT','--CHECK', required=False, default=9999, type=int, help='Checkpoint')
    parser.add_argument('--Load','--L', required=False, default=False, type=bool, help='Choosing Load weight or not [True/False] | Default : False')


    return parser

def cal_epoch(cfg,STEP):
    with open(cfg.TRAIN.ANNOT_PATH, 'r' ) as f:
        lines=len(f.readlines())
    return int(cfg.TRAIN.BATCH_SIZE*STEP/lines)
