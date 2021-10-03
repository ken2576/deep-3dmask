import os
import subprocess

colmap = 'C:\\Users\\Ken\\Downloads\\COLMAP-dev-windows\\COLMAP.bat'

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
def run_colmap(basedir, match_type, dense=False):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')
    
    feature_extractor_args = [
        colmap, 'feature_extractor', 
            '--database_path', os.path.join(basedir, 'database.db'), 
            '--image_path', basedir,
    ]
    feat_output = ( subprocess.check_output(feature_extractor_args, universal_newlines=True) )
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        colmap, match_type, 
            '--database_path', os.path.join(basedir, 'database.db'), 
    ]

    match_output = ( subprocess.check_output(exhaustive_matcher_args, universal_newlines=True) )
    logfile.write(match_output)
    print('Features matched')
    
    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    triangulator_args = [
        colmap, 'point_triangulator',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', basedir,
            '--input_path', os.path.join(basedir, 'sparse', '0'),
            '--output_path', os.path.join(basedir, 'sparse', '0'),
    ]

    map_output = ( subprocess.check_output(triangulator_args, universal_newlines=True) )
    logfile.write(map_output)
    print('Sparse map created')

    if dense:

        if not os.path.exists(os.path.join(basedir, 'dense', '0')):
            os.makedirs(os.path.join(basedir, 'dense', '0'))

        undistorter_args = [
            colmap, 'image_undistorter',
                '--image_path', basedir,
                '--input_path', os.path.join(basedir, 'sparse', '0'),
                '--output_path', os.path.join(basedir, 'dense', '0')
        ]

        map_output = ( subprocess.check_output(undistorter_args, universal_newlines=True) )
        logfile.write(map_output)

        patch_match_args = [
            colmap, 'patch_match_stereo',
                '--workspace_path', os.path.join(basedir, 'dense', '0')
        ]

        map_output = ( subprocess.check_output(patch_match_args, universal_newlines=True) )
        logfile.write(map_output)

        stereo_fusion_args = [
            colmap, 'stereo_fusion',
                '--workspace_path', os.path.join(basedir, 'dense', '0'),
                '--output_path', os.path.join(basedir, 'dense', '0', 'fused.ply')
        ]

        map_output = ( subprocess.check_output(stereo_fusion_args, universal_newlines=True) )
        logfile.write(map_output)

        print('Dense map created')


    logfile.close()
    
    print( 'Finished running COLMAP, see {} for logs'.format(logfile_name) )


