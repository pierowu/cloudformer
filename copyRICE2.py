import os
import numpy as np
import shutil


def mycopyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile, dstpath + fname)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstpath + fname))


import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, './auxiliary/'))
print(dir_name)
dir_name = '/root/autodl-tmp/uformer512_log/'

import argparse
import options

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
print(opt)

import utils

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.determinist = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import CharbonnierLoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data

######### Logs dir ###########
log_dir = os.path.join(dir_name, 'log', opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().__format__("%Y-%d-%m_%H-%M-%S") + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}

# 随机划分
dataset = get_training_data(opt.train_dir, img_options_train)
# train_size = int((1 - 0.2) * len(dataset))
# validation_size = len(dataset) - train_size
new_dataset, test_dataset = torch.utils.data.random_split(dataset, [588, 148])

train_size = int((1 - 0.2) * len(new_dataset))
validation_size = len(new_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(new_dataset, [train_size, validation_size])

new_train_list = '/root/autodl-tmp/RICE2_seed/train/'
new_val_list = '/root/autodl-tmp/RICE2_seed/val/'
new_test_list = '/root/autodl-tmp/RICE2_seed/test/'

print('trainsize=', len(train_dataset))
print('valsize=', len(val_dataset))
print('testsize=', len(test_dataset))

for i in train_dataset.indices:
    mycopyfile(train_dataset.dataset.clean_filenames[i], new_train_list + 'groundtruth/')
    mycopyfile(train_dataset.dataset.noisy_filenames[i], new_train_list + 'input/')

for i in val_dataset.indices:
    mycopyfile(val_dataset.dataset.clean_filenames[i], new_val_list + 'groundtruth/')
    mycopyfile(val_dataset.dataset.noisy_filenames[i], new_val_list + 'input/')

for i in test_dataset.indices:
    mycopyfile(val_dataset.dataset.clean_filenames[i], new_test_list + 'groundtruth/')
    mycopyfile(val_dataset.dataset.noisy_filenames[i], new_test_list + 'input/')