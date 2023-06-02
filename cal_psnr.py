import glob
import os
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

import numpy as np
import cv2

def compare_psnr(img1, img2, maxvalue):
    img1, img2 = img1.astype(np.float64), img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    return 10 * np.log10((maxvalue ** 2) / mse)


path1='/root/autodl-tmp/RICE1spa_test/ground_truth'
path2='/root/autodl-tmp/spa-gan_results/epoch_0001/'

files1 = glob.glob(os.path.join(path1,'*.png'))
files2 = glob.glob(os.path.join(path2,'*.png'))
psnr_val_rgb = []
ssim_val_rgb = []
for file in files1:
    img_name=os.path.split(file)[-1]
    print(img_name)
    img1=cv2.imread(file)
    img2=cv2.imread(path2+img_name)
    psnr_val_rgb.append(psnr_loss(img1, img2))
    ssim_val_rgb.append(ssim_loss(img1, img2, multichannel=True))


psnr_val_rgb = sum(psnr_val_rgb)/len(psnr_val_rgb)
ssim_val_rgb = sum(ssim_val_rgb)/len(ssim_val_rgb)
print("PSNR: %f, SSIM: %f " %(psnr_val_rgb,ssim_val_rgb))





