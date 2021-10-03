import os
import glob
import random
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset
import torchvision.transforms.functional as tf

def data_augmentation(images):
    mode = np.random.randint(0, 5)
    if mode == 0:
        # random brightness
        brightness_factor = 1.0 + random.uniform(-0.2, 0.3)
        xi = tf.adjust_brightness(images, brightness_factor)
    elif mode == 1:
        # random saturation
        saturation_factor = 1.0 + random.uniform(-0.2, 0.5)
        xi = tf.adjust_saturation(images, saturation_factor)
    elif mode == 2:
        # random hue
        hue_factor = random.uniform(-0.2, 0.2)
        xi = tf.adjust_hue(images, hue_factor)
    elif mode == 3:
        # random contrast
        contrast_factor = 1.0 + random.uniform(-0.2, 0.4)
        xi = tf.adjust_contrast(images, contrast_factor)
    else:
        xi = images
    return xi

def random_resized_crop_and_adjust_intrinsics(
    images, intrinsics, crop_h, crop_w,
    min_scale=1, max_scale=1):
    """Randomly resize and crop images, and adjust intrinsics accordingly.
    Args:
        images: [..., H, W] images
        intrinsics: [..., 3, 3] camera intrinsics
        crop_h: (int) height of output crops
        crop_w: (int) width of output crops
        min_scale: (float) minimum scale factor
        max_scale: (float) maximum scale factor
    Returns:
        Randomly resized and cropped images and according intrinsics
    """
    im_size = torch.tensor(images.shape[-2:])
    if min_scale == 1.0 and max_scale == 1.0:
        scale_factor = 1
    else:
        scale_factor = np.random.uniform(min_scale, max_scale)

    scaled_size = (scale_factor * im_size).int()
    offset_limit = scaled_size - torch.tensor([crop_h, crop_w]) + 1
    offset_y = np.random.randint(0, offset_limit[0])
    offset_x = np.random.randint(0, offset_limit[1])

    cropped_images, cropped_intrinsics = crop_image_and_adjust_intrinsics(
        images, intrinsics, scale_factor,
        offset_y, offset_x, crop_h, crop_w)
    return cropped_images, cropped_intrinsics

def crop_image_and_adjust_intrinsics(
    image, intrinsics, scale, offset_y, offset_x,
    crop_h, crop_w):
    """Resize, crop images and adjust instrinsics accordingly.
    Args:
        image: [..., H, W] images
        intrinsics: [..., 3, 3] camera intrinsics
        scale: scale factor for resizing
        offset_y: y-offset in pixels from top of image
        offset_x: x-offset in pixels from left of image
        crop_h: height of region to be cropped
        crop_w: width of region to be cropped
    Returns:
        [..., crop_h, crop_w] cropped images,
        [..., 3, 3] adjusted intrinsics
    """
    im_size = image.shape[-2:]
    resized_images = tf.resize(image, (int(scale*im_size[0]), int(scale*im_size[1])))
    cropped_images = tf.crop(resized_images,
                offset_y, offset_x, crop_h, crop_w)

    cropped_intrinsics = intrinsics.clone()
    cropped_intrinsics[..., :2, :] *= scale
    
    cropped_intrinsics[..., :2, -1] -= torch.tensor([offset_x, offset_y])

    return cropped_images, cropped_intrinsics

def random_subsequence(seq, length, min_stride=1, max_stride=1):
    """Returns a random subsequence with min_stride <= stride <= max_stride.
    For example if self.length = 4 and we ask for a length 2
    sequence (with default min/max_stride=1), there are three possibilities:
    [0,1], [1,2], [2,3].
    Args:
        seq: list of image sequence indices
        length: the length of the subsequence to be returned.
        min_stride: the minimum stride (> 0) between elements of the sequence
        max_stride: the maximum stride (> 0) between elements of the sequence
    Returns:
        A random, uniformly chosen subsequence of the requested length
        and stride.
    """
    # First pick a stride.
    if max_stride == min_stride:
      stride = min_stride
    else:
      stride = np.random.randint(min_stride, max_stride+1)

    # Now pick the starting index.
    # If the subsequence starts at index i, then its final element will be at
    # index i + (length - 1) * stride, which must be less than the length of
    # the sequence. Therefore i must be less than maxval, where:
    maxval = len(seq) - (length - 1) * stride
    start = np.random.randint(0, maxval)
    end = start + 1 + (length - 1) * stride
    return seq[start:end:stride]

class RealEstateDataset(Dataset):
    def __init__(self, data_path, filename,
            num_frames=10,
            min_stride=3,
            max_stride=10,
            num_sources=2,
            split='train'):
        self.data_path = data_path
        self.filename = filename
        self.num_frames = num_frames
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.num_sources = num_sources
        self.split = split
        self.define_transforms()
        self._init_dataset()

    def _init_dataset(self):

        self.img_path = os.path.join(self.data_path, self.filename)
        
        filename = os.path.split(self.img_path)[-1]
        w, h = [int(x) for x in filename[:-3].split('x')]
        self.w = w
        self.h = h

        all_ints = np.load(os.path.join(self.data_path, 'int.npy'), allow_pickle=True)
        all_ints = [np.array(x) for x in all_ints]
        all_exts = np.load(os.path.join(self.data_path, 'ext.npy'), allow_pickle=True)
        all_exts = [np.array(x) for x in all_exts]
        # self.total_scene_count = len(all_ints)

        self.usable_scenes = []
        required_length = (self.num_frames - 1) * self.max_stride + 1
        self.ints = []
        self.exts = []
        for idx, (i, e) in enumerate(zip(all_ints, all_exts)):
            assert len(i) == len(e)
            if len(i) > required_length:
                self.usable_scenes.append(idx)
                self.ints.append(i)
                self.exts.append(e)

        self.total_scene_count = len(self.ints)

        for scene in self.ints:
            # Setup poses            
            scene[:, :1, :] *= w
            scene[:, 1:2, :] *= h

        self.total_img_count = sum([len(x) for x in self.ints])
        print(f'Read {self.total_scene_count} scenes, {self.total_img_count} images')

    def define_transforms(self):
        self.transforms = T.ToTensor()

    def __getitem__(self, idx):
        # Select input/output frame
        img_count = len(self.ints[idx])
        indices = [x for x in range(img_count)]
        subseq = random_subsequence(indices, self.num_frames, self.min_stride, self.max_stride)
        subseq = np.random.permutation(subseq)
        src_cams = subseq[:self.num_sources]
        tgt_cam = subseq[self.num_sources]
        # Read data
        with h5py.File(self.img_path, 'r') as hf:
            scene_idx = self.usable_scenes[idx]
            src_imgs = [self.transforms(hf[str(scene_idx)][x])[:3] for x in src_cams]
            src_imgs = torch.stack(src_imgs, 0)
            tgt_img = self.transforms(hf[str(scene_idx)][tgt_cam])[:3]
        
        src_exts = [torch.FloatTensor(self.exts[idx][x]) for x in src_cams]
        src_exts = torch.stack(src_exts, 0)
        tgt_ext = torch.FloatTensor(self.exts[idx][tgt_cam])

        src_ints = [torch.FloatTensor(self.ints[idx][x]) for x in src_cams]
        src_ints = torch.stack(src_ints, 0)
        tgt_int = torch.FloatTensor(self.ints[idx][tgt_cam])

        if self.split == 'train':
            img_seq = torch.cat([src_imgs, tgt_img.unsqueeze(0)], 0)
            img_ints = torch.cat([src_ints, tgt_int.unsqueeze(0)], 0)
            # TODO remove this
            new_seq, new_ints = random_resized_crop_and_adjust_intrinsics(
                img_seq, img_ints, 256, 256, 1, 2)

            new_seq = data_augmentation(new_seq)

            src_imgs, tgt_img = torch.split(new_seq, (self.num_sources, 1))
            src_ints, tgt_int = torch.split(new_ints, (self.num_sources, 1))

            tgt_img = tgt_img[0]
            tgt_int = tgt_int[0]

        return dict(
            src_imgs=src_imgs,
            src_exts=src_exts,
            src_ints=src_ints,
            tgt_ext=tgt_ext,
            tgt_int=tgt_int,
            tgt_rgb=tgt_img
        )

    def __len__(self):
        return self.total_scene_count