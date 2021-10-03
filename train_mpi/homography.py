import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def divide_safe(num, den):
    eps = 1e-8
    den += eps * torch.eq(den, 0).type(torch.float)
    return num / den

def bilinear_wrapper(imgs, coords):
    '''Wrapper around bilinear sampling function
    
    Args:
        imgs: images to sample [..., height_s, width_s, #channels]
        coords: pixel location to sample from [..., height_t, width_t, 2]
    Returns:
        Sampled images [..., height_t, width_t, #channels]
    '''
    init_dims = imgs.shape[:-3:]
    end_dims_img = imgs.shape[-3::]
    end_dims_coords = coords.shape[-3::]
    prod_init_dims = init_dims[0]
    for i in range(1, len(init_dims)):
        prod_init_dims *= init_dims[i]
    src = imgs.contiguous().view((prod_init_dims,) + end_dims_img)
    coords = coords.view((prod_init_dims,) + end_dims_coords)
    src = src.permute([0, 3, 1, 2])
    tgt = F.grid_sample(src, coords, align_corners=False)
    tgt = tgt.permute([0, 2, 3, 1])
    tgt = tgt.view(init_dims + tgt.shape[-3::])
    return tgt

def inv_homography(k_s, k_t, rot, t, n_hat, a):
    '''Compute inverse homography matrix between two cameras via a plane.
    
    Args:
        k_s: source camera intrinsics [..., 3, 3]
        k_t: target camera intrinsics [..., 3, 3]
        rot: relative roation [..., 3, 3]
        t: translation from source to target camera [..., 3, 1]
        n_hat: plane normal w.r.t source camera frame [..., 1, 3]
        a: plane equation displacement [..., 1, 1]        
    Returns:
        Inverse homography matrices (mapping from target to source)
        [..., 3, 3]
    '''
    rot_t = rot.transpose(-2, -1)
    k_t_inv = torch.inverse(k_t)
    
    denom = a - torch.matmul(torch.matmul(n_hat, rot_t), t)
    numerator = torch.matmul(torch.matmul(torch.matmul(rot_t, t),
                                          n_hat),
                             rot_t)
    inv_hom = torch.matmul(
            torch.matmul(k_s, rot_t + divide_safe(numerator, denom)),
            k_t_inv)
    return inv_hom

def transform_points(points, homography):
    '''Transforms input points according to the homography.
    
    Args:
        points: pixel coordinates [..., height, width, 3]
        homography: transformation [..., 3, 3]
    Returns:
        Transformed coordinates [..., height, width, 3]
    '''
    orig_shape = points.shape
    points_reshaped = points.view(orig_shape[:-3] +
                                  (-1,) +
                                  (orig_shape[-1:]))
    dim0 = len(homography.shape) - 2
    dim1 = dim0 + 1
    transformed_points = torch.matmul(points_reshaped,
                                      homography.transpose(dim0, dim1))
    transformed_points = transformed_points.view(orig_shape)
    return transformed_points

def normalize_homogenous(points):
    '''Converts homogenous coordinates to euclidean coordinates.
    
    Args:
        points: points in homogenous coordinates [..., #dimensions + 1]
    Returns:
        Points in standard coordinates after dividing by the last entry
        [..., #dimensions]
    '''
    uv = points[..., :-1]
    w = points[..., -1].unsqueeze(-1)
    return divide_safe(uv, w)

def transform_plane_imgs(imgs, pixel_coords, k_s, k_t, rot, t, n_hat, a):
    '''Transforms input images via homographies for corresponding planes.
    
    Args:
        imgs: input images [..., height_s, width_t, #channels]
        pixel_coords: pixel coordinates [..., height_t, width_t, 3]
        k_s: source camera intrinsics [..., 3, 3]
        k_t: target camera intrinsics [..., 3, 3]
        rot: relative rotation [..., 3, 3]
        t: translation from source to target camera [..., 3, 1]
        n_hat: plane normal w.r.t source camera frame [..., 1, 3]
        a: plane equation displacement [..., 1, 1]
    Returns:
        Images after bilinear sampling from the input.
    '''
    tgt2src = inv_homography(k_s, k_t, rot, t, n_hat, a)
    pixel_coords_t2s = transform_points(pixel_coords, tgt2src)
    pixel_coords_t2s = normalize_homogenous(pixel_coords_t2s)
    pixel_coords_t2s[..., 0] = pixel_coords_t2s[..., 0] /\
                               pixel_coords_t2s.shape[-2] * 2 - 1
    pixel_coords_t2s[..., 1] = pixel_coords_t2s[..., 1] /\
                               pixel_coords_t2s.shape[-3] * 2 - 1
    imgs_s2t = bilinear_wrapper(imgs, pixel_coords_t2s)
    
    return imgs_s2t

def planar_transform(src_imgs, pixel_coords, k_s, k_t, rot, t, n_hat, a):
    '''Transforms images, masks and depth maps according to 
       planar transformation.

    Args:
        src_imgs: input images [layer, batch, height_s, width_s, #channels]
        pixel_coords: coordinates of target image pixels
                      [batch, height_t, width_t, 3]
        k_s: source camera intrinsics [batch, 3, 3]
        k_t: target camera intrinsics [batch, 3, 3]
        rot: relative rotation [batch, 3, 3]
        t: translation from source to target camera [batch, 3, 1]
        n_hat: plane normal w.r.t source camera frame [layer, batch, 1, 3]
        a: plane equation displacement [layer, batch, 1, 1]
    Returns:
        Images projected to target frame [layer, height, width, #channels]
    '''
    layer = src_imgs.shape[0]
    rot_rep_dims = [layer]
    rot_rep_dims += [1 for _ in range(len(k_s.shape))]
    
    cds_rep_dims = [layer]
    cds_rep_dims += [1 for _ in range(len(pixel_coords.shape))]

    k_s = k_s.repeat(rot_rep_dims)
    k_t = k_t.repeat(rot_rep_dims)
    t = t.repeat(rot_rep_dims)
    rot = rot.repeat(rot_rep_dims)
    pixel_coords = pixel_coords.repeat(cds_rep_dims)
    
    tgt_imgs = transform_plane_imgs(
            src_imgs, pixel_coords, k_s, k_t, rot, t, n_hat, a)
    
    return tgt_imgs