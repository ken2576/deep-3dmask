import os
import glob
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py

from pose2numpy import proc_poses
from dataset_utils import prune_dataset
from get_triplets import proc_folders

def create_img_dataset(folders, out_folder, width, height, postfix=''):
    save_name = os.path.join(out_folder,
        f'{width}x{height}{postfix}.h5')

    with h5py.File(save_name, 'w') as h5f:

        for idx, folder in tqdm(enumerate(folders)):
            im_arr = read_folder(folder, width=width, height=height)
            
            dataIn = h5f.create_dataset(str(idx),
                im_arr.shape,
                np.uint8,
                chunks=(1,) + im_arr.shape[1:],
                compression="lzf", shuffle=True)

            for im_idx, im in enumerate(im_arr):
                dataIn[im_idx] = im

def read_folder(folder, width=None, height=None):
    im_arr = []
    for im_path in sorted(glob.glob(os.path.join(folder, '*.png'))):
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if width and height:
            im = cv2.resize(im, (width, height), cv2.INTER_AREA)
        im_arr.append(im)

    return np.stack(im_arr)

def read_data(path, scene_idx, img_idx):
    with h5py.File(path, 'r') as hf:
        im = hf[str(scene_idx)][img_idx]
    return im

def test_read():
    data = 'small_dataset_50\\640x360.h5'
    triplets = 'small_dataset_50\\triplets.npy'
    pts = np.load(triplets, allow_pickle=True)
    scene_idx = 0
    img_idx = 0
    
    im = read_data(data, scene_idx, img_idx)
    h, w, _ = im.shape

    pts = pts[scene_idx][img_idx]
    pts[0] *= w
    pts[1] *= h

    plt.imshow(im)
    plt.scatter(pts[0], pts[1])
    plt.show()

def proc_data():
    path = '/mnt/Data/RealEstate10K/frames/train'
    txt_folder = '/home/ken/RealEstate10K/train'
    out_folder = '/media/ken/Backup/realestate_train_3500'
    os.makedirs(out_folder, exist_ok=True)
    
    # Excluding some files
    excluded_files = ['00821987dfce6103']
    img_folders = prune_dataset(path, txt_folder, excluded_files)
    np.save('img_folders.npy', img_folders)
    
    # img_folders = np.load('img_folders.npy', allow_pickle=True)
    img_folders = img_folders
    
    # Generate poses
    proc_poses(img_folders, txt_folder, out_folder, getshape=False)

    # Generate image dataset
    create_img_dataset(img_folders, out_folder, 640, 360)

def proc_test():
    path = 'D:\\Project\\RealEstate10k\\frames\\test'
    txt_folder = 'D:\\Project\\RealEstate10k\\test'
    out_folder = 'D:\\Project\\RealEstate10k\\test_dataset_10'
    os.makedirs(out_folder, exist_ok=True)
    
    # Excluding some files
    excluded_files = []
    img_folders = prune_dataset(path, txt_folder, excluded_files)
    np.save('img_folders.npy', img_folders)
    
    # img_folders = np.load('img_folders.npy', allow_pickle=True)
    img_folders = img_folders[:10]
    
    # Generate poses
    proc_poses(img_folders, txt_folder, out_folder, getshape=False)
    
    # Generate triplets
    proc_folders(img_folders, out_folder)

    # Generate image dataset
    create_img_dataset(img_folders, out_folder, 640, 360)


# proc_test()
proc_data()