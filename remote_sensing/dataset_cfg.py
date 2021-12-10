import os
import sys

cfg = {
    # slc-selector
    'orient'        : 0,        # (bool) 0 or 1
    'perc_data'     : .2,       # (float) percentage of slc stripes for either 0 or 1 orient
    
    # sar-preproc
    'pol'           : ['HH', 'HV', 'VH', 'VV'],
    'ml_filter'     : 'avg',        # 'avg', 'med'
    'ml_size'       : 2,            # (int) multi-look kernel: 2, 4, 8
    'in_dir'        : '../../dataset/sn6-expanded', # '../../expanded-dataset', # where the SLC stripes located
    'sar_proc'      : 1,            # (int) num of process used, 1 proc needs up to 7GB RAM
    
    # tiling
    'splits'        : ['train'],  # (list) what split to generate
    # if load_tile is 1, below data is where to load tiles from. Else, it's where to save data
    'project'       : 'sensor',
    'name'          : 'base',       # (str) DO NOT USE 'train', 'val' or 'test' in the name
    'stride'        : 0,            # (int) 0, 80 unit in px
    'out_dir'       : '../../dataset/sensor', # '../../sensor',  # where the tiles will be stored
    'label_dir'     : '../../dataset/spacenet6-challenge/expanded/exp_geojson_buildings', # '../../expanded/geojson_buildings',  
    'tile_proc'     : 4,            # (int) num of process for tiling
    'load_scheme'   : 0,            # (bool) 1 means load from tile_scheme folder instead of tiling from scratch, will only tile raster
    'verbose'       : 0,            # (int) 0 minimum logs, 1 for all info, 2 for necessary tiling
}

post_cfg = {
    # post-tiling - used to create tfrecord
    # will read cfg from tile_dir instead of using cfg above
    'tile_dir'      : '../../dataset/sensor/base/raster',  # where to load tiles and cfg for tfrec
    'splits'        : ['train'],  # which split to create tfrecord
    'tfrec_dir'     : 'tfrecords',   # folder to save tfrecords, change with post-tile versions
    'tfrec_proc'    : 2,            # num of process used, 1 for serial
    'load_fix'      : 0,            # 1 means load from vector_fix instead of vector
    'channel'       : [1,4,3],
    'out_precision' : 32,            # 8, 16 or 32
    'tfrec_size'    : 7,          # num examples per tfrec
    'rotation'      : 0,
    'flip'          : 0,
}

def check_path(save_path):
    if os.path.isdir(save_path):
        user_in = input(f'directory to save: {save_path}, already exist and might contain data, proceed? (y/n)\n')
        if user_in == 'n':
            print('terminating')
            sys.exit()
        elif user_in == 'y':
            print('proceeding')
        else:
            print('only accept "y" or "n", terminating')
            sys.exit()

def print_cfg():
    if cfg['load_scheme']:
        print('loading tiles from scheme')
    else:
        print('creating tiles from scratch')

"""
folder structure
{out_dir}
    {name}
        raster
            cfg.json
        {tfrec_dir}
            post_cfg.json
    s0
        tile_scheme
        vector
        vector_fix
    s80
        tile_scheme
        vector
        vector_fix
    output.tif
    output_1.tif  # if sar_proc > 1

total slc:
    102 of orient 0
    100 of orient 1
so num of slc is just perc_data * 100
    perc_data is percentage for either orient 0 or 1
"""