import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_ckpt
from module import MPINet3d, MaskNet3d
from projector import (divide_safe, over_composite,
        projective_forward_warp, batch_inverse_warp)

class RenderNet(nn.Module):
    def __init__(self, args):
        super(RenderNet, self).__init__()
        
        self.num_depths = 32

        # Load pretrained MPI network
        self.mpi_net = MPINet3d(6, 2) # Try two views first
        load_ckpt(self.mpi_net, args.mpi_model, model_name='models.mpi_net')
        self.mpi_net.eval()

        # Create 3D mask volume network
        self.mask_net = MaskNet3d(12, 1)

    def forward(self, sample):
        src_imgs = sample['src_rgb'] # (batch, #views, #frames, #channels, height, width)
        src_imgs = src_imgs.squeeze(2) # Don't care about multiple frames at the moment
        src_exts = sample['src_w2c']
        src_ints = sample['src_K']
        tgt_ext = sample['tgt_w2c']
        tgt_int = sample['tgt_K']
        bds = sample['bd']
        src_bg = sample['src_bg'] # (batch, #views, #channels, height, width)

        b, v, c, h, w = src_imgs.shape

        depths = self.get_depths(bds[..., 0], bds[..., 1], device=src_imgs.device)

        fg_alphas = []
        rgbs = []
        mask_volumes = []
        for b_idx in range(b):
            # Prepare input
            imgs_to_warp = torch.cat([src_imgs[b_idx], src_bg[b_idx]], 0)
            exts = torch.cat([src_exts[b_idx], src_exts[b_idx]], 0)
            ref_ext = src_exts[b_idx, 0] # First camera is the reference camera
            ints = torch.cat([src_ints[b_idx], src_ints[b_idx]], 0)
            ref_int = src_ints[b_idx, 0] # First camera is the reference camera
            depths_to_warp = depths[..., b_idx]

            psv = self.create_psv(imgs_to_warp,
                                exts,
                                ref_ext,
                                ints,
                                ref_int,
                                depths_to_warp)
            
            psv = psv.reshape([1, -1, self.num_depths, h, w])
            
            with torch.no_grad():
                mpis = self.gen_mpi(self.mpi_net, torch.cat([psv[:, :v*3], psv[:, v*3:]], 0))
            fg_mpi = mpis[:, 0:1].detach()
            bg_mpi = mpis[:, 1:2].detach()
            mask = self.gen_mask(self.mask_net, psv)
            masked_fg = fg_mpi * mask
            masked_bg = bg_mpi * (1-mask)

            full_mpi = masked_fg + masked_bg
            
            warped_mpi = self.warp_mpi(
                torch.cat([full_mpi, masked_fg], -1),
                ref_ext[None, ...],
                ref_int[None, ...],
                tgt_ext[b_idx, None],
                tgt_int[b_idx, None],
                depths[..., b_idx:b_idx+1]
            )

            rgb, _, _, _ = over_composite(warped_mpi[..., :4],
                divide_safe(1., depths[..., b_idx:b_idx+1]))
            rgb = rgb.permute([0, 3, 1, 2])

            _, _, fg_acc_alpha, _ = over_composite(warped_mpi[..., 4:],
                divide_safe(1., depths[..., b_idx:b_idx+1]))

            fg_alphas.append(fg_acc_alpha)
            rgbs.append(rgb)
            mask_volumes.append(mask.permute([1, 4, 0, 2, 3]))

        fg_alphas = torch.cat(fg_alphas, 0) # (batch, height, width)
        rgbs = torch.cat(rgbs, 0) # (batch, 3, height, width)
        mask_volumes = torch.cat(mask_volumes, 0)

        return dict(rgb=rgbs, alpha=fg_alphas, volume=mask_volumes)

    def get_depths(self, min_depth, max_depth, device=None):
        depths = []
        for mind, maxd in zip(min_depth, max_depth):
            depth_arr = 1 / torch.linspace(1./maxd, 1./mind,
                    steps=self.num_depths, device=device)
            depths.append(depth_arr)
        depths = torch.stack(depths, 1)
        return depths

    def create_psv(self, imgs, exts, ref_ext, ints, ref_int, depths):
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
    
        psv = self.batch_warp(psv_input, exts, ints,
            ref_ext[None, :], ref_int[None, :], psv_depths) - 1. # Move back to correct range
        return psv

    def batch_warp(self, src_imgs, src_exts, src_ints, tgt_exts, tgt_ints, depths):
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

    def gen_mask(self, net, psv):
        '''Generate mask volume from plane sweep volume

        Args:
            psv: plane sweep volume [batch, #views * #channels, #planes, height, width]
        Returns:
            Mask volume [#planes, batch, height, width, 1]
        '''
        mask = net(psv)

        return mask.permute([2, 0, 3, 4, 1])

    def gen_mpi(self, net, psv):
        '''Generate multiplane images from plane sweep volume

        Args:
            psv: plane sweep volume [batch, #views * #channels, #planes, height, width]
        Returns:
            Multiplane images [#planes, batch, height, width, 4]
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

    def warp_mpi(self, mpi, ref_ext, ref_int, tgt_ext, tgt_int, depths):
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
        return warped_mpis