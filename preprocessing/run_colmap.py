import os
import glob
import collections
import numpy as np
import matplotlib.pyplot as plt
from llff.poses.colmap_read_model import rotmat2qvec
from colmap_wrapper import run_colmap


Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = '# Camera list with one line of data per camera:\n'
    '#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n'
    '# Number of cameras: {}\n'.format(len(cameras))
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")

def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items()))/len(images)
    HEADER = '# Image list with two lines of data per image:\n'
    '#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n'
    '#   POINTS2D[] as (X, Y, POINT3D_ID)\n'
    '# Number of images: {}, mean observations per image: {}\n'.format(len(images), mean_observations)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")

def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items()))/len(points3D)
    HEADER = '# 3D point list with one line of data per point:\n'
    '#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n'
    '# Number of points: {}, mean track length: {}\n'.format(len(points3D), mean_track_length)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")

def write_model(cameras, images, path):
    write_cameras_text(cameras, os.path.join(path, "cameras.txt"))
    write_images_text(images, os.path.join(path, "images.txt"))
    write_points3D_text({}, os.path.join(path, "points3D.txt"))
    return cameras, images


def create_colmap_cam(idx, width, height, focal_length, cx, cy, model='SIMPLE_PINHOLE'):
    params = np.array([focal_length, cx, cy])
    return Camera(id=idx, model=model,
        width=width, height=height,
        params=params)

def create_colmap_img(idx, name, ext):
    R = ext[:3, :3]
    qvec = rotmat2qvec(R)
    tvec = np.asarray(ext[:3, 3]).squeeze()
    return BaseImage(id=idx, qvec=qvec, tvec=tvec, camera_id=idx, name=name, xys=[], point3D_ids=[])

def read_cam(file):
    with open(file) as f:
        data = f.read().split("\n")
        url = data[0]
        cam_params = np.array([line.split(" ") for line in data[1:-1]]).astype(float)
    return (url, cam_params)

def parse_params(cam_params):
    fx, fy, cx, cy = cam_params[1:5]
    ext = cam_params[7:].reshape([3, 4])
    return fx, fy, cx, cy, ext

def get_shape(folder):
    im_paths = sorted(glob.glob(os.path.join(folder, '*.png')))
    im = plt.imread(im_paths[0])[..., :3]
    return im.shape[:2]

def re2colmap(txt_path, scenedir):
    out_folder = os.path.join(scenedir, 'sparse', '0')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    url, params = read_cam(txt_path)
    imgdir = scenedir
    h, w = get_shape(imgdir)

    colmap_cams = {}
    colmap_imgs = {}

    for idx, cam in enumerate(params):
        fx, fy, cx, cy, ext = parse_params(cam)

        focal_length = fx * w
        cx = cx * w
        cy = cy * h
        colmap_cam = create_colmap_cam(idx+1, w, h, focal_length, cx, cy)

        ext_mat = np.eye(4)
        ext_mat[:3] = ext
        colmap_img = create_colmap_img(idx+1, str(idx).zfill(5) + '.png', ext_mat)
        colmap_cams[idx+1] = colmap_cam
        colmap_imgs[idx+1] = colmap_img

    write_model(colmap_cams, colmap_imgs, out_folder)
    run_colmap(scenedir, 'sequential_matcher')

if __name__ == '__main__':

    # camtxt = 'H:\\RealEstate10K\\test\\0043978734eec081.txt'
    # img_folder = 'H:\\RealEstate10K\\0043978734eec081'

    folders = sorted(glob.glob(os.path.join('H:\\RealEstate10K', 'frames\\test\\*')))
    txt_folder = 'H:\\RealEstate10K\\test'

    for folder in folders:
        name = os.path.split(folder)[-1]
        txt = os.path.join(txt_folder, name + '.txt')
        re2colmap(txt, folder)