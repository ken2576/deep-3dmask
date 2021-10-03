import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_cam(file):
    with open(file) as f:
        data = f.read().split("\n")
        url = data[0]
        cam_params = np.array([line.split(" ") for line in data[1:-1]]).astype(float)
    return (url, cam_params)

def parse_params(cam_params, h, w):
    fx, fy, cx, cy = cam_params[1:5]
    E = np.eye(4)
    E[:3] = cam_params[7:].reshape([3, 4])
    # print(h, w)
    K = np.array([
        [w * fx, 0, w*cx],
        [0, h * fy, h*cy],
        [0,      0,    1]
    ])

    return K, E

def get_shape(folder):
    im_paths = sorted(glob.glob(os.path.join(folder, '*.png')))
    im = plt.imread(im_paths[0])[..., :3]
    return im.shape[:2]

def proc_poses(img_folders, txt_folder, out_folder, getshape=True):

    good_scenes = []
    scene_K_arr = []
    scene_E_arr = []
    for im in img_folders:
        img_count = len(glob.glob(os.path.join(im, '*.png')))
        txt_path = os.path.join(txt_folder, os.path.split(im)[-1] + '.txt')
        _, cams = read_cam(txt_path)

        if getshape:
            h, w = get_shape(im)
        else:
            h = 1
            w = 1

        K_arr = []
        E_arr = []
        for cam in cams:
            K, E = parse_params(cam, h, w)
            K_arr.append(K)
            E_arr.append(E)

        scene_K_arr.append(K_arr)
        scene_E_arr.append(E_arr)
        good_scenes.append(im)
    
    np.save(os.path.join(out_folder, 'int.npy'), np.array(scene_K_arr, dtype=object))
    np.save(os.path.join(out_folder, 'ext.npy'), np.array(scene_E_arr, dtype=object))
    np.save(os.path.join(out_folder, 'scenes.npy'), np.array(good_scenes, dtype=object))

    total_img_count = sum([len(x) for x in scene_E_arr])
    print(f'Processed {len(good_scenes)} scenes, {total_img_count} images')