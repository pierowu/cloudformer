import cv2
import os
import numpy as np
import shutil
def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))

def mymovefile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.move(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))



# train_list_file='./data/RICE_DATASET/RICE1/train_list.txt'
# test_list_file='./data/RICE_DATASET/RICE1/test_list.txt'
# cloud_path='./data/RICE_DATASET/RICE1/cloudy_image'
# true_path='./data/RICE_DATASET/RICE1/ground_truth'
# train_imlist=np.loadtxt(train_list_file,str)
# test_imlist=np.loadtxt(test_list_file,str)
# train_path='./data/RICE_DATASET/RICE1_train/'
# test_path='./data/RICE_DATASET/RICE1_test/'
# for file_name in train_imlist:
#     mycopyfile(os.path.join(cloud_path,file_name),train_path+'cloudy_image/')
#     mycopyfile(os.path.join(true_path, file_name), train_path + 'ground_truth/')
# for file_name in test_imlist:
#     mycopyfile(os.path.join(cloud_path,file_name),test_path+'cloudy_image/')
#     mycopyfile(os.path.join(true_path, file_name), test_path + 'ground_truth/')
# print('hhh')

train_path='./datasets/RICE210_840/train/'
val_path='./datasets/RICE210_840/val/'
file_list=os.listdir(train_path+'input')
eval_sample_nums=40
val_file_list=[]
for i in range(eval_sample_nums):
    randIndex = int(np.random.uniform(0, len(file_list)))    # 获得0~len(eval_sample_nums)的一个随机数
    val_file_list.append(file_list[randIndex])
    val_file_list.append(file_list[randIndex])
    del(file_list[randIndex])
    del(file_list[randIndex])

for file_name in val_file_list:
    mymovefile(train_path+'input/'+file_name,val_path+'input/')
    mymovefile(train_path + 'groundtruth/' + file_name, val_path + 'groundtruth/')
