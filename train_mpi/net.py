import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import MPINet3d
from projector import (divide_safe, over_composite, projective_forward_warp, batch_inverse_warp)

class RenderNet(nn.Module):
    def __init__(self):
        super(RenderNet, self).__init__()
        self.num_depths = 32
        self.mpi_net = MPINet3d(6, 2) # Try two views first

    def forward(self, sample):
        src_imgs = sample['src_imgs'] # (batch, #views, #channels, height, width)
        src_exts = sample['src_exts']
        src_ints = sample['src_ints']
        tgt_ext = sample['tgt_ext']
        tgt_int = sample['tgt_int']

        b, v, c, h, w = src_imgs.shape
        
        if 'bd' in sample:
            bds = sample['bd']
            # TODO HACK Does not handle batch size more than 1
            depths = self.get_depths(bds[0, 0], bds[0, 1], device=src_imgs.device)[:, None]
        else:
            depths = self.get_depths(device=src_imgs.device)[:, None]
        psv_depths = depths.expand([-1, b*v])

        psv_input = src_imgs.reshape([-1, c, h, w]) + 1.
        psv_src = src_exts.reshape([-1, 4, 4])
        psv_tgt = src_exts[:, 0:1].repeat([1, v, 1, 1]).reshape([-1, 4, 4])
        psv_src_int = src_ints.reshape([-1, 3, 3])
        psv_tgt_int = src_ints[:, 0:1].repeat([1, v, 1, 1]).reshape([-1, 3, 3])

        psv = self.create_psv(psv_input, psv_src, psv_src_int,
            psv_tgt, psv_tgt_int, psv_depths)
        # psv (batch, #channels, #planes, height, width)
        psv = psv.reshape([b, v * 3, self.num_depths, h, w]) - 1 # a trick to deal with out of FoV regions

        # Create MPI
        pred_ = self.mpi_net(psv)
        
        weights = pred_[:, 1:]
        alpha = pred_[:, :1]
        rgb_layers = torch.zeros_like(psv[:, :3])
        for v_idx in range(v):
            rgb_layers += weights[:, v_idx:v_idx+1] * psv[:, 3*v_idx:3*(v_idx+1)]

        rgba_layers = torch.cat([rgb_layers, alpha], 1)
        rgba_layers = rgba_layers.permute([2, 0, 3, 4, 1])
        transforms = torch.matmul(tgt_ext, torch.inverse(src_exts[:, 0]))

        warped_mpis = projective_forward_warp(rgba_layers, src_ints[:, 0], tgt_int,
            transforms, depths)

        rendering, _, acc_alpha, disp = over_composite(warped_mpis, divide_safe(1., depths))
        results = dict(rgb=rendering.permute([0, 3, 1, 2]), acc_alpha=acc_alpha, disp=disp)

        return results

    def get_depths(self, min_depth=1, max_depth=100, device=None):
        depths = 1./torch.linspace(1./max_depth, 1./min_depth, self.num_depths, device=device)
        return depths

    def create_psv(self, src_imgs, src_exts, src_ints, tgt_exts, tgt_ints, depths):
        '''Create PSV from inputs
        Args:
            src_imgs: source images [batch, #channels, height, width]
            src_exts: source extrinsics [batch, 4, 4]
            src_ints: source intrinsics [batch, 3, 3]
            tgt_exts: target extrinsics [batch, 4, 4]
            tgt_ints: target intrinsics [batch, 3, 3]
            depths: depth values [#planes, batch]
        Returns:
            Plane sweep volume [batch, #channels, #depth_planes, height, width]
        '''
        trnfs = torch.matmul(src_exts, torch.inverse(tgt_exts))

        psv = batch_inverse_warp(src_imgs, depths,
                src_ints, torch.inverse(tgt_ints), trnfs)

        return psv