
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

from visualizer import get_local
get_local.activate()
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/wangzd/uformer/')

import scipy.io as sio
from utils.loader import get_validation_data
import utils

from model import UNet,Uformer,Uformer_Cross,Uformer_CatCross

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss


def grid_show(to_shows, cols):
    rows = max((len(to_shows) - 1) // cols + 1,2)
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 8,rows * 8 ))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            # axs[i].imshow(image)
            # axs[i].set_title(title)
            # axs[i].set_yticks([])
            # axs[i].set_xticks([])
            # axs[j].imshow(image)
            # axs[j].set_title(title)
            # axs[j].set_yticks([])
            # axs[j].set_xticks([])
    f = plt.gcf()
    plt.show()
    plt.close()
    return f



# a=np.array([[255,0],[0,100]])
# b=[a,a]
# grid_show(b,2)


def visualize_head(att_map):
    plt.figure(figsize=(6.4, 6.4))
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()
    f = plt.gcf()
    return f

def visualize_head_full(att_map):
    # plt.figure(figsize=(6.4, 6.4))
    # ax = plt.gca()
    # Plot the heatmap
    # im = ax.imshow(att_map)
    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax)
    h, w = att_map.shape
    plt.imshow(att_map)
    plt.axis('off')
    plt.gcf().set_size_inches(w / 100.0 , h / 100.0 )
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)


    # plt.imshow(att_map)
    # plt.axis('off')
    f = plt.gcf()
    return f


def visualize_heads(att_map, cols):
    to_shows = []
    # att_map = att_map.squeeze()
    for i in range(att_map.shape[0]):
        to_shows.append((att_map[i], f'Head {i}'))
    average_att_map = att_map.mean(axis=0)
    to_shows.append((average_att_map, 'Head Average'))
    f=grid_show(to_shows, cols=cols)
    return f


def gray2rgb(image):
    return np.repeat(image[..., np.newaxis], 3, 2)


def cls_padding(image, mask, cls_weight, grid_size):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    image = np.array(image)

    H, W = image.shape[:2]
    delta_H = int(H / grid_size[0])
    delta_W = int(W / grid_size[1])

    padding_w = delta_W
    padding_h = H
    padding = np.ones_like(image) * 255
    padding = padding[:padding_h, :padding_w]

    padded_image = np.hstack((padding, image))
    padded_image = Image.fromarray(padded_image)
    draw = ImageDraw.Draw(padded_image)
    draw.text((int(delta_W / 4), int(delta_H / 4)), 'CLS', fill=(0, 0, 0))  # PIL.Image.size = (W,H) not (H,W)

    mask = mask / max(np.max(mask), cls_weight)
    cls_weight = cls_weight / max(np.max(mask), cls_weight)

    if len(padding.shape) == 3:
        padding = padding[:, :, 0]
        padding[:, :] = np.min(mask)
    mask_to_pad = np.ones((1, 1)) * cls_weight
    mask_to_pad = Image.fromarray(mask_to_pad)
    mask_to_pad = mask_to_pad.resize((delta_W, delta_H))
    mask_to_pad = np.array(mask_to_pad)

    padding[:delta_H, :delta_W] = mask_to_pad
    padded_mask = np.hstack((padding, mask))
    padded_mask = padded_mask

    meta_mask = np.zeros((padded_mask.shape[0], padded_mask.shape[1], 4))
    meta_mask[delta_H:, 0: delta_W, :] = 1

    return padded_image, padded_mask, meta_mask


def visualize_grid_to_grid_with_cls(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    attention_map = att_map[grid_index]
    cls_weight = attention_map[0]

    mask = attention_map[1:].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    padded_image, padded_mask, meta_mask = cls_padding(image, mask, cls_weight, grid_size)

    if grid_index != 0:  # adjust grid_index since we pad our image
        grid_index = grid_index + (grid_index - 1) // grid_size[1]

    grid_image = highlight_grid(padded_image, [grid_index], (grid_size[0], grid_size[1] + 1))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(padded_mask, alpha=alpha, cmap='rainbow')
    ax[1].imshow(meta_mask)
    ax[1].axis('off')


def visualize_grid_to_grid(att_map, grid_index, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    H, W = att_map.shape
    with_cls_token = False

    grid_image = highlight_grid(image, [grid_index], grid_size)

    mask = att_map[grid_index].reshape(grid_size[0], grid_size[1])
    mask = Image.fromarray(mask).resize((image.size))

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    fig.tight_layout()

    ax[0].imshow(grid_image)
    ax[0].axis('off')

    ax[1].imshow(grid_image)
    ax[1].imshow(mask / np.max(mask), alpha=alpha, cmap='rainbow')
    ax[1].axis('off')
    plt.show()


def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a = ImageDraw.ImageDraw(image)
        a.rectangle([(y * w, x * h), (y * w + w, x * h + h)], fill=None, outline='red', width=2)
    return image

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='../uformer/datasets/denoising/sidd/test/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/denoising/sidd/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/Uformer32_0520_v2_irpe_ratio/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
args = parser.parse_args()


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

model_restoration= utils.get_arch(args)
model_restoration = torch.nn.DataParallel(model_restoration)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()
with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        get_local.clear()
        rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
        rgb_noisy = data_test[1].cuda()
        filenames = data_test[2]

        rgb_restored = model_restoration(rgb_noisy)

        rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
        psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
        ssim_val_rgb.append(ssim_loss(rgb_restored, rgb_gt, multichannel=True))

        cache = get_local.cache
        print(list(cache.keys()))
        attention_maps = cache['WindowAttention.forward']
        # for i in range(len(attention_maps)):
        #     # attn=np.mean(attention_maps[i],axis=0)
        #     attention_map=attention_maps[i]
        #     attn=attention_map[attention_map.shape[0]//2]
        #     f=visualize_heads(attn, cols=max(attn.shape[0]//2,2))
        #     try:
        #         f.savefig(r'./attn_windowmid/layer{}/pic{}_layer{}.png'.format(i,ii,i))
        #     except FileNotFoundError:
        #         if not os.path.exists(r'./attn_windowmid/layer{}'.format(i)):
        #             os.makedirs(r'./attn_windowmid/layer{}'.format(i))
        #         f.savefig(r'./attn_windowmid/layer{}/pic{}_layer{}.png'.format(i, ii, i))
        #
        #     f.clear()  # 释放内存

        # for i in [0,1,2,10]:
        #     attn=np.mean(attention_maps[i],axis=0)
        #     # attention_map=attention_maps[i]
        #     # attn=attention_map[attention_map.shape[0]//2]
        #     attn=np.mean(attn,axis=0)
        #     f=visualize_head(attn)
        #     try:
        #         f.savefig(r'./attn_ave/layer{}/pic{}_layer{}.png'.format(i,ii,i))
        #     except FileNotFoundError:
        #         if not os.path.exists(r'./attn_ave/layer{}'.format(i)):
        #             os.makedirs(r'./attn_ave/layer{}'.format(i))
        #         f.savefig(r'./attn_ave/layer{}/pic{}_layer{}.png'.format(i, ii, i))
        #
        #     f.clear()  # 释放内存

        for i in [0,10]:
            attn=np.mean(attention_maps[i],axis=0)
            # attention_map=attention_maps[i]
            # attn=attention_map[attention_map.shape[0]//2]
            attn=np.mean(attn,axis=0)
            f=visualize_head_full(attn)
            try:
                f.savefig(r'./attn_ave/layer{}/pic{}_layer{}.png'.format(i,ii,i))
            except FileNotFoundError:
                if not os.path.exists(r'./attn_ave/layer{}'.format(i)):
                    os.makedirs(r'./attn_ave/layer{}'.format(i))
                f.savefig(r'./attn_ave/layer{}/pic{}_layer{}.png'.format(i, ii, i),dpi=300)

            f.clear()  # 释放内存






        if args.save_images:
            utils.save_img(os.path.join(args.result_dir,filenames[0]), img_as_ubyte(rgb_restored))

psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
print("PSNR: %f, SSIM: %f " %(psnr_val_rgb,ssim_val_rgb))

