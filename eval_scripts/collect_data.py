import os
import glob
import shutil
import argparse

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Collect data for mask net.')
    parser.add_argument('--root_dir', type=str,
                        help='data root directory')
    parser.add_argument('--range', nargs='+',
                        type=int,
                        default=[0, 60],
                        help='range for frames to use')
    parser.add_argument('--bg_skip', type=int,
                        default=25,
                        help='background frame skip')
    parser.add_argument('--no_bg', action='store_true',
                        help='no background')
    parser.add_argument('--img_wh', nargs='+',
                        type=int,
                        default=[640, 360],
                        help='image width and height')
    parser.add_argument('--original_size', action='store_true',
                        help='if true, store original size in images directory (for LLFF)')
    parser.add_argument('--extension', type=str,
                        default=".jpg",
                        help='image extension (default: .png)')
    parser.add_argument('--out_dir', type=str,
                        default="frames_to_render",
                        help='output directory name')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    width, height = args.img_wh

    cams = sorted(glob.glob(os.path.join(args.root_dir, 'cam*', '')))
    if not args.no_bg:
        bg_folder = os.path.join(args.root_dir, 'background')
        os.makedirs(bg_folder, exist_ok=True)

        img_count = len(glob.glob(os.path.join(cams[0], f'*{args.extension}')))
        bg_frames = [x for x in range(img_count)]
        bg_frames = bg_frames[::args.bg_skip]

        for idx, cam in enumerate(cams):
            img_paths = sorted(glob.glob(os.path.join(cam, f'*{args.extension}')))
            imgs_to_use = [img_paths[x] for x in bg_frames]
            if args.extension == '.jpg':
                imgs = [plt.imread(x)/255. for x in imgs_to_use]
            else:
                imgs = [plt.imread(x) for x in imgs_to_use]
            imgs = np.stack(imgs)
            bg = np.median(imgs, 0)
            bg *= 255.
            bg = Image.fromarray(np.uint8(bg))
            # if width and height:
            #     bg = bg.resize((width, height), Image.LANCZOS)
            bg.save(os.path.join(bg_folder, f'cam{idx:02d}.png'))

    start, end = args.range
    pose_path = os.path.join(args.root_dir, 'poses_bounds.npy')
    out_dir = os.path.join(args.root_dir, args.out_dir)

    for frame in range(start, end):
        dst_folder = os.path.join(out_dir, f'{frame:05d}')
        
        if args.original_size:
            image_folder = 'images'
        else:
            image_folder = f'images_{width}x{height}'
        
        os.makedirs(os.path.join(dst_folder, image_folder), exist_ok=True)
        
        for idx, cam in enumerate(cams):
            src = os.path.join(cam, f'{frame:05d}{args.extension}')
            dst = os.path.join(dst_folder, image_folder, f'{idx:03d}{args.extension}')

            im = Image.open(src)
            if not args.original_size:
                im = im.resize((width, height), Image.LANCZOS)
            im.save(dst)
            pose_dst = os.path.join(dst_folder, 'poses_bounds.npy')
            shutil.copy2(pose_path, pose_dst)