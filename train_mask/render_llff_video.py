import os
import glob

import torch
import numpy as np
from tqdm import tqdm
import imageio
import configargparse
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from threading import Thread

from utils import load_ckpt, move_to
from utils.visualization import save_rgba, save_psv
import metrics

from module import MPINet3d, MaskNet3d
from projector import (divide_safe, over_composite, batch_inverse_warp, projective_forward_warp)

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument('--root_dir', type=str,
                        help='root directory of video frames')
    parser.add_argument('--bg_dir', type=str,
                        help='root directory of background images')
    parser.add_argument('--out_dir', type=str,
                        help='output directory')

    parser.add_argument('--indices', nargs='+', type=int, default=[4, 5],
                        help='camera indices')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[640, 360],
                        help='resolution (img_h, img_w) of the image')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    return parser.parse_args()

class ImageSequenceWriter:
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        self.index = 0
        os.makedirs(path, exist_ok=True)
        
    def add_batch(self, frames):
        Thread(target=self._add_batch, args=(frames, self.index)).start()
        self.index += len(frames)
            
    def _add_batch(self, frames, index):
        for i in range(len(frames)):
            frame = frames[i]
            frame = Image.fromarray(frame)
            frame.save(os.path.join(self.path, str(index + i).zfill(5) + '.' + self.extension))

def pose2mat(pose):
    """Convert pose matrix (3x5) to extrinsic matrix (4x4) and
       intrinsic matrix (3x3)
    
    Args:
        pose: 3x5 pose matrix
    Returns:
        Extrinsic matrix (4x4 w2c) and intrinsic matrix (3x3)
    """
    extrinsic = np.eye(4)
    extrinsic[:3, :] = pose[:, :4]
    inv_extrinsic = np.linalg.inv(extrinsic)
    extrinsic = np.linalg.inv(inv_extrinsic)
    h, w, focal_length = pose[:, 4]
    intrinsic = np.array([[focal_length, 0, w/2],
                        [0, focal_length, h/2],
                        [0,            0,   1]])

    return extrinsic, intrinsic

def convert_llff(pose):
    """Convert LLFF poses to PyTorch convention (w2c extrinsic and hwf)
    """
    hwf = pose[:3, 4:]

    ext = np.eye(4)
    ext[:3, :4] = pose[:3, :4]
    mat = np.linalg.inv(ext)
    mat = mat[[1, 0, 2]]
    mat[2] = -mat[2]
    mat[:, 2] = -mat[:, 2]

    return np.concatenate([mat, hwf], -1)

def load_poses(filename):
    
    if filename.endswith('npy'):
        return np.load(filename)
    
    elif filename.endswith('txt'):
        with open(filename, 'r') as file:
            file.readline()
            x = np.loadtxt(file)
        x = np.transpose(np.reshape(x, [-1,5,3]), [0,2,1])
        x = np.concatenate([-x[...,1:2], x[...,0:1], x[...,2:]], -1)
        return x
    
    print('Incompatible pose file {}, must be .txt or .npy'.format(filename))
    return None

def transform():
    return T.Compose(
        [T.ToTensor()]
    )

def read_bg_data(folder, img_wh=(640, 360)):
    trfs = transform()

    img_paths = sorted(glob.glob(os.path.join(folder, '*.png'))) + \
        sorted(glob.glob(os.path.join(folder, '*.jpg')))
    src_bgs = [Image.open(x) for x in img_paths]
    src_bgs = [bg.resize(img_wh, Image.LANCZOS) for bg in src_bgs]
    src_bgs = torch.stack([trfs(x) for x in src_bgs])

    return src_bgs


def read_llff_data(rootdir, img_wh=(640, 360)):
    trfs = transform()

    poses_bounds = np.load(os.path.join(rootdir, 'poses_bounds.npy'))
    bds = poses_bounds[:, -2:]
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
    new_poses = [convert_llff(x) for x in poses]
    src_w2cs = np.array([pose2mat(x)[0] for x in new_poses])
    src_ints = np.array([pose2mat(x)[1] for x in new_poses])

    img_paths = sorted(glob.glob(os.path.join(rootdir, f'images', '*.png'))) + \
        sorted(glob.glob(os.path.join(rootdir, f'images', '*.jpg'))) 
    src_imgs = [Image.open(x) for x in img_paths]
    src_imgs = [im.resize(img_wh, Image.LANCZOS) for im in src_imgs]
    src_imgs = torch.stack([trfs(x) for x in src_imgs])

    src_ints[:, :2] *= img_wh[0] / src_ints[:, 0:1, -1:] / 2 # Scale image accordingly

    src_w2cs = torch.FloatTensor(src_w2cs)
    src_ints = torch.FloatTensor(src_ints)
    out_bds = torch.FloatTensor([np.min(bds) *.9, np.max(bds) * 2])

    return src_imgs, src_w2cs, src_ints, out_bds

def read_tgt_pose(filename, img_wh=(640, 360)):
    poses = load_poses(filename)
    new_poses = [convert_llff(x) for x in poses]
    tgt_w2cs = np.array([pose2mat(x)[0] for x in new_poses])
    tgt_ints = np.array([pose2mat(x)[1] for x in new_poses])

    tgt_ints[:, :2] *= img_wh[0] / tgt_ints[:, 0:1, -1:] / 2 # Scale image accordingly

    tgt_w2cs = torch.FloatTensor(tgt_w2cs)
    tgt_ints = torch.FloatTensor(tgt_ints)
    return tgt_w2cs, tgt_ints

@torch.no_grad()
def inference(mpi_net, mask_net, sample, num_depths=32):  
    src_imgs = sample['src_rgb']
    src_bg = sample['src_bg']
    src_ints = sample['src_K']
    src_exts = sample['src_w2c']
    bds = sample['bd']

    depths = get_depths(bds[0], bds[1], device=src_imgs.device)

    v, c, h, w = src_imgs.shape
    imgs_to_warp = torch.cat([src_imgs, src_bg], 0)
    exts = torch.cat([src_exts, src_exts], 0)
    ref_ext = src_exts[0] # First camera is the reference camera
    ints = torch.cat([src_ints, src_ints], 0)
    ref_int = src_ints[0] # First camera is the reference camera

    psv = create_psv(imgs_to_warp,
                    exts,
                    ref_ext,
                    ints,
                    ref_int,
                    depths)
    psv = psv.reshape([1, -1, num_depths, h, w])

    mpis = gen_mpi(mpi_net, torch.cat([psv[:, :v*3], psv[:, v*3:]], 0))
    fg_mpi = mpis[:, 0:1]
    bg_mpi = mpis[:, 1:2]
    mask = gen_mask(mask_net, psv)

    # plt.imsave('src.png', src_imgs[0].squeeze().permute([1, 2, 0]).cpu().numpy())
    # exit()

    masked_fg = fg_mpi * mask
    masked_bg = bg_mpi * (1-mask)

    full_mpi = masked_fg + masked_bg
    _, _, fg_acc_alpha, _ = over_composite(masked_fg, depths.unsqueeze(1))

    return dict(mpi=full_mpi, fg_mpi=fg_mpi, depth=depths.unsqueeze(1))

def get_depths(min_depth, max_depth, num_depths=32, device=None):
    depths = 1 / torch.linspace(1./max_depth, 1./min_depth,
            steps=num_depths, device=device)
    return depths

def create_psv(imgs, exts, ref_ext, ints, ref_int, depths):
    '''Create plane sweep volume from inputs

    Args:
        imgs: source images [#views, #channels, height, width]
        exts: source extrinsics [#views, 4, 4]
        ref_ext: reference extrinsics [4, 4]
        ints: source intrinsics [#views, 3, 3]
        ref_int: reference intrinsics [3, 3]
        depths: depth values [#planes]
    Returns:
        Plane sweep volume [#views, #channels, #depth_planes, height, width]
    '''
    num_views = imgs.shape[0]
    psv_depths = depths.unsqueeze(1).repeat([1, num_views])
    psv_input = imgs + 1.

    psv = batch_warp(psv_input, exts, ints,
        ref_ext[None, :], ref_int[None, :], psv_depths) - 1. # Move back to correct range
    return psv

def batch_warp(src_imgs, src_exts, src_ints, tgt_exts, tgt_ints, depths):
    '''Warp images to target pose

    Args:
        src_imgs: source images [batch, #channels, height, width]
        src_exts: source extrinsics [batch, 4, 4]
        src_ints: source intrinsics [batch, 3, 3]
        tgt_exts: target extrinsics [batch, 4, 4]
        tgt_ints: target intrinsics [batch, 3, 3]
        depths: depth values [#planes, batch]
    Returns:
        Warped images [batch, #channels, #depth_planes, height, width]
    '''
    trnfs = torch.matmul(src_exts, torch.inverse(tgt_exts))

    psv = batch_inverse_warp(src_imgs, depths,
            src_ints, torch.inverse(tgt_ints), trnfs)

    return psv

def gen_mask(net, psv):
    '''Generate mask volume from plane sweep volume

    Args:
        psv: plane sweep volume [batch, #views * #channels, #planes, height, width]
    Returns:
        Mask volume [batch, 1, #planes, height, width]
    '''
    mask = net(psv)

    return mask.permute([2, 0, 3, 4, 1])

def gen_mpi(net, psv):
    '''Generate multiplane images from plane sweep volume

    Args:
        psv: plane sweep volume [batch, #views * #channels, #planes, height, width]
    Returns:
        Multiplane images [batch, 4, #planes, height, width]
    '''
    num_channels = psv.shape[1]
    pred_ = net(psv)

    weights = pred_[:, 1:]
    alpha = pred_[:, :1]
    rgb_layers = torch.zeros_like(psv[:, :3])
    for v_idx in range(num_channels//3):
        rgb_layers += weights[:, v_idx:v_idx+1] * psv[:, 3*v_idx:3*(v_idx+1)]

    rgba_layers = torch.cat([rgb_layers, alpha], 1)
    rgba_layers = rgba_layers.permute([2, 0, 3, 4, 1])
    return rgba_layers

@torch.no_grad()
def render_mpi(mpi, ref_ext, ref_int, tgt_ext, tgt_int, depths):
    '''Render image from multiplane images

    Args:
        mpi: multiplane images [layers, batch, height, width, #channels]
        ref_ext: reference pose [batch, 4, 4]
        ref_int: reference intrinsics [batch, 3, 3]
        tgt_ext: target pose [batch, 4, 4]
        tgt_int: target intrinsics [batch, 3, 3]
        depths: depth values [#planes, batch]
    '''
    transforms = torch.matmul(tgt_ext, torch.inverse(ref_ext))
    warped_mpis = projective_forward_warp(mpi, ref_int, tgt_int,
        transforms, depths)

    rendering, _, _, _ = over_composite(warped_mpis, divide_safe(1., depths))
    return rendering.permute([0, 3, 1, 2])

if __name__ == "__main__":
    device = 'cuda'

    args = get_opts()

    mask_folder = os.path.join(args.out_dir, 'mask')
    os.makedirs(mask_folder, exist_ok=True)

    ext = 'png'
    mask_writer = ImageSequenceWriter(mask_folder, ext)

    mpi_net = MPINet3d(6, 2).to(device)
    load_ckpt(mpi_net, args.ckpt_path, model_name='models.mpi_net')
    mpi_net.eval()

    mask_net = MaskNet3d(12, 1).to(device)
    load_ckpt(mask_net, args.ckpt_path, model_name='models.mask_net')
    mask_net.eval()

    src_bgs = read_bg_data(args.bg_dir)
    src_bg = src_bgs[args.indices].to(device)

    frames = sorted(glob.glob(os.path.join(args.root_dir, '[0-9][0-9][0-9]*', '')))
    num_frames = len(frames)

    tgt_w2cs, tgt_ints = read_tgt_pose(os.path.join(frames[0], 'qual_path.txt'))
    tgt_ints = tgt_ints.to(device)
    tgt_w2cs = tgt_w2cs.to(device)

    for i in tqdm(range(len(tgt_w2cs))):
        tgt_w2c = tgt_w2cs[i:i+1]
        tgt_int = tgt_ints[i:i+1]

        folder_num = i % num_frames
        frame = frames[folder_num]
        src_imgs, src_w2cs, src_ints, bds = read_llff_data(frame)
        src_imgs = src_imgs[args.indices].to(device)
        src_w2cs = src_w2cs[args.indices].to(device)
        src_ints = src_ints[args.indices].to(device)
        bds = bds.to(device)

        sample=dict(
            src_rgb=src_imgs,
            src_bg=src_bg,
            src_K=src_ints,
            src_w2c=src_w2cs,
            bd=bds
        )
        results = inference(mpi_net, mask_net, sample)
        mpi = results['mpi']
        depths = results['depth']
        fg_mpi = results['fg_mpi']

        rendering = render_mpi(
            mpi,
            src_w2cs[0, None, ...], src_ints[0, None, ...],
            tgt_w2c, tgt_int, depths
        )

        img_pred = rendering.squeeze().permute([1, 2, 0]).cpu().numpy()
        img_pred = np.clip(img_pred, 0.0, 1.0)
        img_pred_ = (img_pred*255).astype(np.uint8)
        mask_writer.add_batch(img_pred_[None, :])

        torch.cuda.empty_cache()
