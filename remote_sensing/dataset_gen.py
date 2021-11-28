# input: SN6-expanded dataset
# preprocess SAR according to dataset-scheme
# saves to temporary raster 'output.tif' (to minimize storage usage)
# read saved raster and do tiling according to regions
# output tile name: sensor_20190823162315_20190823162606_base_train_100_0016.tif
# project_ timestamp_ dsscheme_ split_ coverage_ tile_id


# import matplotlib.pyplot as plt
# import numpy as np
import os

# import rasterio as rs
# from rasterio import plot
# from rasterio import features as feat  # convert to mask

# import geopandas as gpd

# from shapely.geometry import box
# import pandas as pd

# import glob
import json
import pickle
import time
import sys
from multiprocessing import Pool

from timebudget import timebudget

from dataset_cfg import cfg
# check save path exist or not to prevent overwritting valuable data
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


from sar_preproc import SarPreproc
from tile_gen import get_labels_bounds, raster_vector_tiling
from tile_scheme import load_raster_vector_tiling, load_scheme, parallel_tile_generator
# import lib.tfrec as tfrec

def run_parallel_ops(operation, input, pool):
    pool.starmap(operation, input)

if __name__=='__main__':
    # load all timestamps and their sar orientation
    with open('timestamp_orientation.pickle','rb') as f:
        time_orient = pickle.load(f)

    # split dataset based on each label split
    print('loading labels')
    labels, bounds = get_labels_bounds(cfg["label_dir"])
    print(f'labels loaded')

    # loop through timestamps
    with timebudget('PRE-PROC + TILING'):
        for i,to in enumerate(time_orient[70:]):
            print(f'processing raster {to}.. {i+70} of {len(time_orient)}')
            timestamp = to[:-2]
            orient = to[-1]

            # process SAR
            out_fn = 'output.tif'  # f'{to}.tif'  # give specific output
            sar_preproc = SarPreproc(cfg, timestamp, cfg["in_dir"], cfg["out_dir"], out_fn)
            with timebudget('SAR PRE-PROC'): sar_preproc()
            
            # tile raster and vector
            proc_slc_path = os.path.join(cfg["out_dir"], out_fn)  # output.tif path
            if cfg['load_tile']:
                print('loading tiles from scheme')
                # combining schemes
                schemes = []
                for split in ['train','val','test']:
                    scheme = load_scheme(cfg, timestamp, orient, split)
                    for el in scheme:
                        schemes.append((el, proc_slc_path, save_path))
                
                print(len(schemes))
                proc_pool = Pool(8)
                run_parallel_ops(parallel_tile_generator, schemes, proc_pool)
                # load_raster_vector_tiling(cfg, timestamp, orient, proc_slc_path, save_path)

            else:
                print(f'creating tiles')
                with timebudget('TILING SLC'):
                    raster_dict, vector_dict = raster_vector_tiling(
                        cfg, labels, bounds, timestamp, orient, proc_slc_path, save_path)

# end of code








# CLOSE PARALEL, wait all process to finish

# # write tfrecords
# start = time.time()
# in_path = tile_out_path = os.path.join(cfg["out_dir"], cfg["name"])
# for split in ['train','val','test']:
#     # grab and shuffle [raster, vector] file paths
#     fp = tfrec.get_tile_paths(in_path, split, shuffle=True, perc_data=cfg["perc_data"])
    
#     # create save directory if not exist
#     base_dir = os.path.join(cfg["out_dir"], cfg["name"], cfg["tfrec_dir"])
#     if not os.path.isdir(base_dir):
#         os.makedirs(base_dir)
    
#     # START PARALEL BASED ON fp indexes
#     # name of tfrec file
#     base_fn = os.path.join(base_dir, split)
#     tfrec.create_tfrecord(*fp, cfg, base_fn, cfg["tfrec_size"])

# end = time.time()
# print(f'tfrecords created in {(end-start):.1f}s')

# cfg_fn = os.path.join(base_dir, 'cfg.json')
# with open(cfg_fn, 'w') as f:
#     json.dump(cfg, f)

# print('config saved!')