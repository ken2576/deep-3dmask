#!bin/bash
# Usage:
# sh render.sh [frame_folder] [starting_frame] [ending_frame] [output_folder_name]

s=$2
ns=$(printf %05d $s)

python ../eval_scripts/collect_data.py --root_dir $1 --range $2 $3 --original_size --out_dir $4
python ../eval_scripts/imgs2renderpath.py --scenedir $1/$4/$ns/ --outname $1/$4/$ns/qual_path.txt