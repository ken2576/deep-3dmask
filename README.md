# Deep 3D Mask Volume for View Synthesis of Dynamic Scenes

Official PyTorch Implementation of paper "Deep 3D Mask Volume for View Synthesis of Dynamic Scenes", ICCV 2021.

[Kai-En Lin](https://cseweb.ucsd.edu/~k2lin/)<sup>1</sup>, [Lei Xiao](https://leixiao-ubc.github.io/)<sup>2</sup>, [Feng Liu](http://web.cecs.pdx.edu/~fliu/)<sup>2</sup>, [Guowei Yang](https://www.framiere.com)<sup>1</sup>, [Ravi Ramamoorthi](https://cseweb.ucsd.edu/~ravir/)<sup>1</sup>

<sup>1</sup>University of California, San Diego, <sup>2</sup>Facebook Reality Labs

[Project Page](https://cseweb.ucsd.edu//~viscomp/projects/ICCV21Deep/) | [Paper](https://cseweb.ucsd.edu//~viscomp/projects/ICCV21Deep/assets/deep_iccv.pdf) | [Supplementary Materials](https://cseweb.ucsd.edu//~viscomp/projects/ICCV21Deep/assets/deep_iccv_supp.pdf) | [Pretrained models](https://drive.google.com/drive/u/1/folders/1Yt_lAeMcq6VbK4Cqao-lkGFbxVAU3H4U) | [Dataset](https://drive.google.com/drive/u/1/folders/1KJjCujC_p4cHDXYPmnLUhMI2vfLv7wIY) | [Preprocessing script](https://github.com/ken2576/multiview_preprocessing)

## Requirements

### Install required packages

Make sure you have up-to-date NVIDIA drivers supporting CUDA 11.1 (10.2 could work but need to change ```cudatoolkit``` package accordingly)

Run

```
conda env create -f environments.yml
conda activate video_viewsynth
```
## Usage

### Rendering

0. Download [our pretrained checkpoint](https://drive.google.com/drive/u/1/folders/1Yt_lAeMcq6VbK4Cqao-lkGFbxVAU3H4U) and [testing data](https://drive.google.com/file/d/1_9KA20cI_0Bs9ERkT65TPtiom3fdQXD0/view?usp=sharing). Extract the content to ```[path_to_data_directory]```.
    It contains ```frames``` and ```background``` folders, as well as ```poses_bounds.npy```.

1. In ```configs```, setup data path by changing ```render_video.txt```

    ```root_dir``` should point to the ```frames``` folder mentioned in 1. and ```bg_dir``` should point to ```background``` folder.
    
    ```out_dir``` can be your desired output folder.
    
    ```ckpt_path``` should be the pretrained checkpoint path.

2. Run ```python render_llff_video.py --config [config_file_path]```

    e.g. ```python render_llff_video.py --config ../configs/render_video.txt```


* (Optional) For your own data, please run ```prepare_data.sh```

    ```sh render.sh [frame_folder] [starting_frame] [ending_frame] [output_folder_name]```

    Make sure your data is in this structure before running

    ```
    [frame_folder] --- cam00 --- 00000.jpg
                    |         |- 00001.jpg
                    |         ...
                    |- cam01
                    |- cam02
                    ...
                    |- poses_bounds.npy
    ```

    e.g.  ```sh render.sh ~/deep_3d_data/frames 0 20 qual```

### Training

#### Train MPI

0. Download [RealEstate10K dataset](https://google.github.io/realestate10k/) and extract the frames. There are scripts in ```preprocessing``` folder which can be used to generate the data.

    The order should be ```download_data.py``` -> ```extract_frames.py``` -> ```compress_data.py```.
    
    Remember to change the path in ```compress_data.py```.

1. Change the paths in config file ```train_realestate10k.txt```

2. Run

    ```
    cd train_mpi
    python train.py --config ../configs/train_realestate10k.txt
    ```

#### Train Mask

Once MPI is trained, we can use the checkpoint to train 3D mask network.

0. Download [dataset](https://drive.google.com/drive/u/1/folders/1YJMaCQiY1lU5mFiycsX01TkwZRc11-0t)

1. Change the paths in config file ```train_mask.txt```

2. Run

    ```
    cd train_mask
    python train.py --config ../configs/train_mask.txt
    ```

## Citation

```
@inproceedings {lin2021deep,
    title = {Deep 3D Mask Volume for View Synthesis of Dynamic Scenes},
    author = {Kai-En Lin and Lei Xiao and Feng Liu and Guowei Yang and Ravi Ramamoorthi},
    booktitle = {ICCV},
    year = {2021},
}
```
