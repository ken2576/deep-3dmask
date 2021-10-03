import os, glob
import configargparse
import traceback
import logging

import numpy as np
import youtube_dl

import itertools
from multiprocessing import Pool

def read_txt(file_list):
    vid_data = []
    for file in file_list:
        with open(file) as f:
            data = f.read().split("\n")
            url = data[0]
            cam_params = np.array([line.split(" ") for line in data[1:-1]]).astype(float)
        vid_data.append((url, cam_params))
    return vid_data

def download_video(url, path, filename):
    print('Downloading: ', url, '...')

    if os.path.exists(os.path.join(path, filename + '.mp4')):
        print('Already done! Skipping...')
        return None

    ydl_opts = {
        'format': 'bestvideo/best',
        'noplaylist': True,
        'outtmpl': os.path.join(path, filename + '.mp4'),
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except Exception as e:
        logging.error(traceback.format_exc())
        print("skipping...")

def proc_files(vids, filepaths, outfolder, n_jobs=5):
    urls = [vid[0] for vid in vids]
    filenames = [vid[0].rsplit('=', -1)[-1] for vid in vids]
    
    data = [x for x in zip(urls, itertools.repeat(outfolder), filenames)]
    with Pool(processes=n_jobs) as pool:
        res = pool.starmap_async(download_video, data)
        res.get()

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Downloader for Google RealEstate10K/MannequinChallenge dataset.')
    parser.add_argument('--config', is_config_file=True,
                        required=True,
                        help='configurations for downloading')
    parser.add_argument('--txt_folder', type=str,
                        help='directory to the txt files')
    parser.add_argument('--out_folder', type=str,
                        help='output directory')
    parser.add_argument('--range', nargs='+',
                        default=[0, None],
                        help='ranges of files to download')
    args = parser.parse_args()
    

    txt_folder = args.txt_folder
    out_folder = args.out_folder
    start, end = args.range
    print(args)

    files = sorted(glob.glob(os.path.join(txt_folder, '*.txt')))  
    files = files[start:end]

    vids = read_txt(files)

    postfix = os.path.split(txt_folder)[-1]
    proc_files(vids, files, os.path.join(out_folder, postfix))