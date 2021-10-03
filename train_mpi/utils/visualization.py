import os
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

# TODO change these
def save_rgba(volume, folder):
    '''Save rgba volume to images
    (D, H, W, C)
    '''
    for i, im in enumerate(volume):
        rgb = im[..., :3] * 255.
        rgb = rgb.cpu().numpy().astype(np.uint8)
        plt.imsave(os.path.join(folder, f'rgb_{i:03d}.png'), rgb)
        alpha = torch.clamp(im[..., 3].squeeze(), 0.0, 1.0)
        alpha = alpha.cpu().numpy()
        tmp = torch.clamp(im.squeeze(), 0.0, 1.0)
        plt.imsave(os.path.join(folder, f'alpha_{i:03d}.png'), alpha, vmin=0.0, vmax=1.0)
        plt.imsave(os.path.join(folder, f'rgba_{i:03d}.png'), tmp.cpu().numpy())

def save_psv(volume, folder, n_views):
    '''Save plane sweep volume to images
    '''
    b, c, d, h, w = volume.shape
    psv = volume.permute([2, 0, 1, 3, 4])
    for idx, im in enumerate(psv):
        for v in range(n_views):
            tmp = im[:, 3*v:3*(v+1)] * 255.
            tmp = tmp.permute([0, 2, 3, 1])[0]
            tmp = tmp.cpu().detach().numpy().astype(np.uint8)
            plt.imsave(folder + str(v) + '_' + str(idx) + '.png', tmp)
        
        combine = im.view([b, n_views, int(c/n_views), h, w])
        tmp = torch.mean(combine, 1) * 255.
        tmp = tmp.permute([0, 2, 3, 1])[0]
        tmp = tmp.cpu().detach().numpy().astype(np.uint8)
        plt.imsave(folder + 'combined_' + str(idx) + '.png', tmp)