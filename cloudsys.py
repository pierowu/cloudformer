import cv2
import glob
import os
import random
import numpy as np

# class RGB2RGBCLOUD(BaseTrain):
#     def __init__(self, dir_rgb, dir_cloud, imlist_rgb, *args, **kwargs):
#         super().__init__()
#         self.rgb = read_imlist(dir_rgb, imlist_rgb)
#         self.cloud = list(glob.glob(os.path.join(dir_cloud, '*.png')))
#         self.size = kwargs.pop('size')
#         self.augmentation = kwargs.pop('augmentation')
#
#     def __len__(self):
#         return len(self.rgb)
#
#     def get_example(self, i):
#         rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)
#         cloud = cv2.imread(random.choice(self.cloud), -1).astype(np.float32)
#
#         alpha = cloud[:, :, 3] / 255.
#         alpha = np.broadcast_to(alpha[:, :, None], alpha.shape + (3,))
#         clouded_rgb = (1. - alpha) * rgb + alpha * cloud[:, :, :3]
#         clouded_rgb = np.clip(clouded_rgb, 0., 255.)
#
#         cloud = cloud[:, :, 3]
#         rgbcloud = np.concatenate((rgb, cloud[:, :, None]), axis=2)
#
#         rgb = clouded_rgb.transpose(2, 0, 1) / 127.5 - 1.
#         rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.
#
#         rgb, rgbcloud = self.transform(rgb, rgbcloud)
#
#         return rgb, rgbcloud

class RGB2RGBCLOUD():
    def __init__(self, dir_rgb, dir_cloud, savedir,rgb_cloud_file,*args, **kwargs):
        super().__init__()
        self.rgb = list(glob.glob(os.path.join(dir_rgb, '*.png')))
        self.cloud = list(glob.glob(os.path.join(dir_cloud, '*.png')))
        # self.size = kwargs.pop('size')
        # self.augmentation = kwargs.pop('augmentation')
        self.savedir=savedir
        # self.rgb_cloud_file=rgb_cloud_file
        self.real_cloud_rgb_image = cv2.imread(rgb_cloud_file)

    def __len__(self):
        return len(self.rgb)

    def get_example(self, i):
        rgb = cv2.imread(self.rgb[i], 1).astype(np.float32)
        cloud_mask = cv2.imread(self.cloud[i], -1).astype(np.float32)


        alpha = cloud_mask[:, :] / 255.
        alpha = np.broadcast_to(alpha[:, :, None], alpha.shape + (3,))
        clouded_rgb = (1. - alpha) * rgb + alpha * self.real_cloud_rgb_image[:, :, :3]
        clouded_rgb = np.clip(clouded_rgb, 0., 255.)
        img_name = os.path.split(self.rgb[i])[-1]
        cv2.imwrite(os.path.join(self.savedir,img_name),clouded_rgb)


        # cloud = cloud[:, :, 3]
        # rgbcloud = np.concatenate((rgb, cloud[:, :, None]), axis=2)
        #
        # rgb = clouded_rgb.transpose(2, 0, 1) / 127.5 - 1.
        # rgbcloud = rgbcloud.transpose(2, 0, 1) / 127.5 - 1.
        #
        # rgb, rgbcloud = self.transform(rgb, rgbcloud)

        return
dir_rgb=r'D:\research\Uformer-main\datasets\Data\Paris_dataset800\RGB'
dir_cloud=r'D:\research\Uformer-main\datasets\Data\Paris_dataset800\mask'
savedir=r'D:\research\Uformer-main\datasets\Data\Paris_dataset800\RGBcloud'
rgb_cloud_file=r'D:\research\Uformer-main\datasets\Data\Paris_dataset800\real_cloud_rgb_image.png'
s=RGB2RGBCLOUD(dir_rgb,dir_cloud,savedir,rgb_cloud_file)
print(len(s))
for i in range(len(s)):
    s.get_example(i)