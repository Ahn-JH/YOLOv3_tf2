import numpy as np
import os
import sys
import glob
import re



def max_checkpoint(location):
    num_list=[]
    data_dict={}
    for file in glob.glob(f'{location}/*.index'):
        file_num=file.split('yolov3_')[-1].split('.index')[0]
        file_num=file.split('-')[-1].split('.')[0]

        data_dict[int(file_num)]=file_loss
        num_list.append(int(file_num))
    max_num=np.max(num_list)
    loss=data_dict[max_num]
    return max_num, loss



if __name__=="__main__":
    max_checkpoint('/home/jihun/TF2/SIMULATION')



