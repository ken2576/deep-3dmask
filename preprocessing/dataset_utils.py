import os
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pose2numpy import read_cam

def check_img():
    path = 'D:\\Project\\RealEstate10k\\metadata\\train'
    scene_paths = np.load(os.path.join(path, 'scenes.npy'), allow_pickle=True)

    bad_images = []
    for scene in tqdm(scene_paths):
        im_paths = sorted(glob.glob(os.path.join(scene, '*.png')))
        for im_path in im_paths:
            # try:
            im = Image.open(im_path)
            im.verify()
            # except PIL.UnidentifiedImageError as e:
            #     bad_images.append(im_path)
            #     print('Bad image detected!')

    print(bad_images)
    # np.save('bad_images.npy', bad_images)

def prune_dataset(path, metadata, excluded_files=[], min_num=50):
    data = []

    t = tqdm(sorted(glob.glob(os.path.join(path, '*', ''))))

    for folder in t:
        file_id = folder.rsplit(os.sep)[-2]
        if file_id in excluded_files:
            print(f'Excluding {file_id}')
            continue
        _, cam = read_cam(os.path.join(metadata, file_id + '.txt'))
        
        imgs = sorted(glob.glob(os.path.join(folder, '*.png')))
        
        if len(cam) == len(imgs):
            if len(imgs) > min_num:
                t.set_description(f'Adding {folder}, images ({len(cam)})')
                data.append(os.path.split(folder)[0])
            # else:
                # print(f'Skipping {folder}, too few images ({len(cam)})')

    return data

if __name__ == '__main__':
    data = prune_dataset()
    print(data)