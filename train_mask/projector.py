import numpy as np
import torch
import torch.nn.functional as F

from homography import planar_transform

def divide_safe(num, denom):
    eps = 1e-8
    tmp = denom + eps * torch.le(denom, 1e-20).to(torch.float)
    return num / tmp

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

def pose2mat(pose):
    """Convert pose matrix (3x5) to extrinsic matrix (4x4) and
       intrinsic matrix (3x3)
    
    Args:
        pose: 3x5 pose matrix
    Returns:
        Extrinsic matrix (4x4) and intrinsic matrix (3x3)
    """
    extrinsic = torch.eye(4)
    extrinsic[:3, :] = pose[:, :4]
    inv_extrinsic = torch.inverse(extrinsic)
    extrinsic = torch.inverse(inv_extrinsic)
    h, w, focal_length = pose[:, 4]
    intrinsic = torch.Tensor([[focal_length, 0, w/2],
                              [0, focal_length, h/2],
                              [0,            0,   1]])

    return extrinsic, intrinsic

def transmittance(alphas):
    '''Convert the alpha maps (ordered from back to front) to transmittance maps.

    Args:
        alphas: alpha maps [#planes, batch, height, width]
    Returns:
        Transmittance maps. [#planes, batch, height, width]
    '''
    # Calculate transmittance
    cum_alphas = torch.cumprod(1.0 - torch.flip(alphas, [0]) + 1.0e-8, 0)
    cum_alphas = torch.cat([torch.ones_like(cum_alphas[0:1]), cum_alphas[:-1]], 0)
    cum_alphas = torch.flip(cum_alphas, [0])

    return cum_alphas * alphas

def over_composite(rgba, dispvals):
    '''Over composite the RGBA (ordered from back to front).
    Args:
        rgba: RGBA layers [#planes, batch, height, width, 4]
        dispvals: disparity values [#plane, batch]
    Returns:
        Rendering: [batch, height, width, 3]
        Accumulated rendering: [#planes, batch, height, width, 3]
        Accumulated alpha: [batch, height, width]
        Disparity map: [batch, height, width]
    '''
    # Calculate transmittance
    rgb = rgba[..., :-1]
    alphas = rgba[..., -1]
    t = transmittance(alphas)
    
    acc_rendering = rgb * t[..., None]
    rendering = torch.sum(acc_rendering, 0)
    acc_alpha = torch.sum(t, 0)
    disp = torch.sum(t * dispvals[..., None, None], 0)

    return rendering, acc_rendering, acc_alpha, disp

def meshgrid_pinhole(h, w,
                    is_homogenous=True, device=None):
    '''Create a meshgrid for image coordinate

    Args:
        h: grid height
        w: grid width
        is_homogenous: return homogenous or not
    Returns:
        Image coordinate meshgrid [height, width, 2 (3 if homogenous)]
    '''
    xs = torch.linspace(0, w-1, steps=w, device=device) + 0.5
    ys = torch.linspace(0, h-1, steps=h, device=device) + 0.5
    new_y, new_x = torch.meshgrid(ys, xs)
    grid = (new_x, new_y)

    if is_homogenous:
        ones = torch.ones_like(new_x)
        grid = torch.stack(grid + (ones, ), 2)
    else:
        grid = torch.stack(grid, 2)
    return grid

def projective_forward_warp(src_imgs, k_s, k_t, transforms, depths):
    '''Forward warp a source image to the target image using homography
    
    Args:
        src_imgs: [layers, batch, height, width, #channels]
        k_s: source intrinsics [batch, 3, 3]
        k_t: target intrinsics [batch, 3, 3]
        transforms: source to target transformation [batch, 4, 4]
        depths: [layers, batch]
    Returns:
        Projected images [layers, batch, height, width, #channels]
    '''
    layer, batch, height, width, _ = src_imgs.shape
    # rot: relative rotation, [batch, 3, 3] matrices
    # t: translations from source to target camera (R*p_s + t = p_t),
    #    [batch, 3, 1]
    # n_hat: plane normal w.r.t source camera frame [layers, batch, 1, 3]
    # a: plane equation displacement (n_hat * p_src + a = 0)
    #    [layers, batch, 1, 1]
    rot = transforms[:, :3, :3]
    t = transforms[:, :3, 3:]
    n_hat = torch.Tensor([0., 0., 1.]).expand(layer, batch, 1, -1).to(src_imgs.device)
    a = -depths.view([layer, batch, 1, 1])
    pixel_coords = meshgrid_pinhole(height, width, device=src_imgs.device).unsqueeze(0)
    pixel_coords = pixel_coords.repeat([batch, 1, 1, 1])
    proj_imgs = planar_transform(
            src_imgs, pixel_coords, k_s, k_t, rot, t, n_hat, a)
    return proj_imgs.clone()

def projective_inverse_warp(src_im, depth, src_int, inv_tgt_int, trnsf, h=-1, w=-1):
    """Projective inverse warping for image

    Args:
        src_im: source image [batch, #channel, height, width]
        depth: depth of the image
        src_int: I_s matrix for source camera [batch, 3, 3]
        inv_tgt_int: I_t^-1 for target camera [batch, 3, 3]
        trnsf: E_s * E_t^-1, the transformation between cameras [batch, 4, 4]
        h: target height
        w: target width
    Returns:
        Warped image
    """
    if h == -1 or w == -1:
        b, _, h, w = src_im.shape
        src_h, src_w = h, w
    else:
        b, _, src_h, src_w = src_im.shape

    # Generate image coordinates for target camera
    im_coord = meshgrid_pinhole(h, w, device=src_im.device)
    coord = im_coord.view([-1, 3])
    coord = coord.unsqueeze(0).repeat([b, 1, 1])

    # Convert to camera coordinates
    cam_coord = torch.matmul(inv_tgt_int.unsqueeze(1), coord[..., None])
    cam_coord = cam_coord * depth[:, None, None, None]
    ones = torch.ones([b, h*w, 1, 1]).to(cam_coord.device)

    cam_coord = torch.cat([cam_coord, ones], 2)

    # Convert to another camera's coordinates
    new_cam_coord = torch.matmul(trnsf.unsqueeze(1), cam_coord)

    # Convert to image coordinates at source camera
    im_coord = torch.matmul(src_int.unsqueeze(1), new_cam_coord[:, :, :3])
    im_coord = im_coord.squeeze(dim=3).view([b, h, w, 3])
    im_coord = im_coord[..., :2] / im_coord[..., 2:3]
    im_coord[..., 0] = im_coord[..., 0] / src_w * 2 - 1.
    im_coord[..., 1] = im_coord[..., 1] / src_h * 2 - 1.

    # Sample from the source image
    warped = F.grid_sample(src_im, im_coord, align_corners=True)
    return warped

def batch_inverse_warp(src_im, depths, src_int, inv_tgt_int, trnsf, h=-1, w=-1):
    """Projective inverse warping for image

    Args:
        src_im: source image [batch, #channel, height, width]
        depths: depths of the image [#planes, batch]
        src_int: I_s matrix for source camera [batch, 3, 3]
        inv_tgt_int: I_t^-1 for target camera [batch, 3, 3]
        trnsf: E_s * E_t^-1, the transformation between cameras [batch, 4, 4]
        h: target height
        w: target width
    Returns:
        Warped image [batch #channel, #planes, height, width]
    """
    if h == -1 or w == -1:
        b, _, h, w = src_im.shape
        src_h, src_w = h, w
    else:
        b, _, src_h, src_w = src_im.shape

    # Generate image coordinates for target camera
    im_coord = meshgrid_pinhole(h, w, device=src_im.device)
    coord = im_coord.view([-1, 3])
    coord = coord.unsqueeze(0).repeat([b, 1, 1])

    # Convert to camera coordinates
    cam_coord = torch.matmul(inv_tgt_int.unsqueeze(1), coord[..., None])
    cam_coord = cam_coord * depths[:, :, None, None, None]
    ones = torch.ones_like(cam_coord[..., 0:1, :])

    cam_coord = torch.cat([cam_coord, ones], -2)

    # Convert to another camera's coordinates
    new_cam_coord = torch.matmul(trnsf.unsqueeze(1), cam_coord)

    # Convert to image coordinates at source camera
    im_coord = torch.matmul(src_int.unsqueeze(1), new_cam_coord[..., :3, :])
    im_coord = im_coord.squeeze(dim=-1).view(depths.shape + (h, w, 3))

    # Fix for NaN and backward projection
    zeros = torch.zeros_like(im_coord[..., 2:3])
    im_coord[..., 2:3] = torch.where(im_coord[..., 2:3] > 0, im_coord[..., 2:3], zeros)
    im_coord = divide_safe(im_coord[..., :2], im_coord[..., 2:3])
    im_coord[..., 0] = im_coord[..., 0] / src_w * 2 - 1.
    im_coord[..., 1] = im_coord[..., 1] / src_h * 2 - 1.

    # Sample from the source image
    warped = F.grid_sample(src_im.repeat(depths.shape[0], 1, 1, 1),
                           im_coord.view(-1, *(im_coord.shape[2:])), align_corners=True)
    return warped.view(depths.shape + (3, h, w)).permute([1, 2, 0, 3, 4])
