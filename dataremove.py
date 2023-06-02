import cv2
import os
import numpy as np
import shutil
import glob
def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, os.path.join(dstpath , fname))         # 复制文件
        # print ("copy %s -> %s"%(srcfile, dstpath + fname))

def mymovefile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.move(srcfile, os.path.join(dstpath , fname))          # 复制文件
        # print ("copy %s -> %s"%(srcfile, dstpath + fname))



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


#paris800
# src_path=r'D:\research\Uformer-main\datasets\Data\Paris_dataset\images\RGB'
# src_path1=r'D:\research\Uformer-main\datasets\Data\Paris_dataset\images\cloud_mask'
# dst_path=r'D:\research\Uformer-main\datasets\Data\Paris_dataset800\RGB'
# dst_path1=r'D:\research\Uformer-main\datasets\Data\Paris_dataset800\mask'
# file_list=list(glob.glob(os.path.join(src_path, '*.png')))
# nums=800
# remove_file_list=[]
# for i in range(nums):
#     randIndex = int(np.random.uniform(0, len(file_list)))    # 获得0~len(eval_sample_nums)的一个随机数
#     remove_file_list.append(file_list[randIndex])
#     remove_file_list.append(file_list[randIndex])
#     del(file_list[randIndex])
#     del(file_list[randIndex])
#
# for file_name in remove_file_list:
#     name=os.path.split(file_name)[-1]
#     mycopyfile(os.path.join(src_path,name),dst_path)
#     mycopyfile(os.path.join(src_path1, name), dst_path1)

src_path1=r'D:\research\Uformer-main\datasets\Data\Paris800\RGB'
src_path=r'D:\research\Uformer-main\datasets\Data\Paris800\RGBcloud'
train_path=r'D:\research\Uformer-main\datasets\Data\Paris800\train'
val_path=r'D:\research\Uformer-main\datasets\Data\Paris800\val'
test_path=r'D:\research\Uformer-main\datasets\Data\Paris800\test'
# dst_path1=r'D:\research\Uformer-main\datasets\Data\Paris_dataset800\mask'
file_list=list(glob.glob(os.path.join(src_path, '*.png')))

num_train=int(len(file_list)*0.64)
num_val=int(len(file_list)*0.16)
num_test=int(len(file_list)*0.2)

val_list=[]
test_list=[]
for i in range(num_test):
    randIndex = int(np.random.uniform(0, len(file_list)))    # 获得0~len(eval_sample_nums)的一个随机数
    test_list.append(file_list[randIndex])

    del(file_list[randIndex])


for i in range(num_val):
    randIndex = int(np.random.uniform(0, len(file_list)))    # 获得0~len(eval_sample_nums)的一个随机数
    val_list.append(file_list[randIndex])

    del(file_list[randIndex])


train_list=file_list

for file_name in test_list:
    name=os.path.split(file_name)[-1]
    mycopyfile(os.path.join(src_path,name),os.path.join(test_path,'input'))
    mycopyfile(os.path.join(src_path1, name), os.path.join(test_path,'groundtruth'))

for file_name in val_list:
    name=os.path.split(file_name)[-1]
    mycopyfile(os.path.join(src_path,name),os.path.join(val_path,'input'))
    mycopyfile(os.path.join(src_path1, name), os.path.join(val_path,'groundtruth'))

for file_name in train_list:
    name = os.path.split(file_name)[-1]
    mycopyfile(os.path.join(src_path, name), os.path.join(train_path, 'input'))
    mycopyfile(os.path.join(src_path1, name), os.path.join(train_path, 'groundtruth'))