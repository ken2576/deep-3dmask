import os, glob
import configargparse
import shutil

import cv2
import numpy as np
from threading import Thread
from PIL import Image

class ImageSequenceWriter:
    def __init__(self, path, extension):
        self.path = path
        self.extension = extension
        self.index = 0
        os.makedirs(path, exist_ok=True)
        
    def add_batch(self, frames):
        Thread(target=self._add_batch, args=(frames, self.index)).start()
        self.index += len(frames)
            
    def _add_batch(self, frames, index):
        for i in range(len(frames)):
            frame = frames[i]
            frame = Image.fromarray(frame)
            frame.save(os.path.join(self.path, str(index + i).zfill(5) + '.' + self.extension))

def create_folder(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            print("Directory creation failed at %s" % path)
        else:
            print("Directory created for %s" % path)

def read_txt(file):
    with open(file) as f:
        data = f.read().split("\n")
        url = data[0]
        cam_params = np.array([line.split(" ") for line in data[1:-1]]).astype(float)
        return (url, cam_params)

def extract_and_save(frame_nums, vid_path, curr_folder):
    cap = cv2.VideoCapture(vid_path)
    print(curr_folder)
    writer = ImageSequenceWriter(curr_folder, 'jpg')
    for frame_num in frame_nums:
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_num/1000)
        ret, frame = cap.read()
        if ret:
            frame = frame[..., ::-1]
            writer.add_batch(frame[None, :])

    cap.release()
    cv2.destroyAllWindows()

def proc_files(filepaths, vid_folder, out_folder):
    for file in filepaths:

        url, cam_params = read_txt(file)
        filename = url.rsplit('=', -1)[-1] + '.mp4'
        frame_nums = cam_params[:, 0]
        vid_path = os.path.join(vid_folder, filename)

        print(vid_path)
        if not glob.glob(vid_path):
            print("No video found")
            continue

        save_filename = file.rsplit(os.sep, -1)[-1].replace('.txt', '')
        curr_folder = os.path.join(out_folder, save_filename)
        if os.path.exists(curr_folder):
            data_count = len([name for name in os.listdir(curr_folder) if os.path.isfile(os.path.join(curr_folder, name))])    
            if data_count == len(frame_nums):
                print("Already done! skipping...")
                continue
            else:
                shutil.rmtree(curr_folder)
                
        extract_and_save(frame_nums, vid_path, curr_folder)
       
if __name__ == '__main__':
    parser = configargparse.ArgumentParser(description='Extracting frames from RealEstate10K/MannequinChallenge dataset.')
    parser.add_argument('--config', is_config_file=True,
                        required=True,
                        help='configurations for downloading')
    parser.add_argument('--txt_folder', type=str,
                        help='directory to the txt files')
    parser.add_argument('--vid_folder', type=str,
                        help='directory to the video files')
    parser.add_argument('--out_folder', type=str,
                        help='output directory')
    parser.add_argument('--range', nargs='+', type=int,
                        default=[0, None],
                        help='ranges of files to download')
    args = parser.parse_args()
    

    txt_folder = args.txt_folder
    vid_folder = args.vid_folder
    out_folder = args.out_folder
    start, end = args.range
    print(args)
    postfix = os.path.split(txt_folder)[-1]

    files = sorted(glob.glob(os.path.join(txt_folder, '*.txt')))
    files = files[start:end]

    proc_files(files, os.path.join(vid_folder, postfix), os.path.join(out_folder, postfix))