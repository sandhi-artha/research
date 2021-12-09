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
    'splits'        : ['train','val','test'],  # (list) what split to generate
    # if load_tile is 1, below data is where to load tiles from. Else, it's where to save data
    'project'       : 'sensor',
    'name'          : 'base_test',
    'stride'        : 0,            # (int) 0, 80 unit in px
    'out_dir'       : '../../dataset/sensor', # '../../sensor',  # where the tiles will be stored
    'label_dir'     : '../../dataset/spacenet6-challenge/expanded/exp_geojson_buildings', # '../../expanded/geojson_buildings',  
    'tile_proc'     : 4,            # (int) num of process for tiling
    'load_scheme'   : 1,            # (bool) 1 means load from tile_scheme folder instead of tiling from scratch, will only tile raster
    'verbose'       : 0,            # (int) 0 minimum logs, 1 for all info, 2 for necessary tiling
}

def check_path():
    save_path = os.path.join(cfg["out_dir"], cfg["name"])
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