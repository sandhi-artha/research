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
from multiprocessing import Pool, current_process

from timebudget import timebudget

from dataset_cfg import cfg, check_path, print_cfg
save_path = os.path.join(cfg["out_dir"], cfg["name"])

from sar_preproc import SarPreproc
from tile_gen import raster_vector_tiling, load_scheme, parallel_tile_generator
# import lib.tfrec as tfrec


def run_parallel_op(ops, args, num_proc, debug=False):
    """determines parallel or serial ops by num_proc
    ops  : function
    args : list of tuples,
        each element of list will be fed to ops
    num_proc : int,
        how many process to create
    debug : bool,
        if True, gives 1 arg to each num_proc, just for testing
    """
    if debug: args = args[:num_proc]
    if num_proc == 1:  # serial, feed all args one by one
        for arg in args: ops(*arg)
    else:  # parallel, let Pool divide args evenly to each proc
        proc_pool = Pool(num_proc)
        proc_pool.starmap(ops, args)

def pre_proc_tiling(timestamp, i, tot):
    print(f'processing raster {timestamp}.. {i} of {tot}')
    orient = cfg['orient']

    if cfg['sar_proc'] > 1:  # if multiproc, write output for each proc so they don't colide
        proc_id = current_process()._identity[0]
        out_fn = f'output_{proc_id}.tif'
    else:
        out_fn = 'output.tif'

    # process SAR
    sar_preproc = SarPreproc(cfg, timestamp, cfg["in_dir"], cfg["out_dir"], out_fn)
    with timebudget('SAR PRE-PROC'): sar_preproc()
    
    # tile raster and vector
    proc_slc_path = os.path.join(cfg["out_dir"], out_fn)  # output.tif path
    with timebudget('TILING'):
        if cfg['load_scheme']:
            args = []
            for split in cfg['splits']:  # combining schemes to list of args
                schemes = load_scheme(cfg, timestamp, orient, split)
                for scheme in schemes:
                    args.append((scheme, proc_slc_path, save_path))
            
            run_parallel_op(parallel_tile_generator, args, cfg['tile_proc'])

        else:
            _, _ = raster_vector_tiling(cfg, timestamp, orient, proc_slc_path, save_path)


if __name__=='__main__':
    # check save path exist or not to prevent overwritting valuable data
    check_path()
    print_cfg()

    # load slc timestamps from pickle, result of dataset_selector.py
    sample_fn = 'sample_{}_{}.pickle'.format(
        int(cfg['perc_data']*100), cfg['orient'])

    with open(sample_fn, 'rb') as f:
        timestamps = pickle.load(f)

    # add idx and len for add arguments in parallel
    args = [(ts,i,len(timestamps)) for i,ts in enumerate(timestamps)]

    with timebudget('PRE-PROC + TILING'):
        run_parallel_op(pre_proc_tiling, args, cfg['sar_proc'], debug=True)
    
    cfg_fn = os.path.join(cfg['out_dir'], cfg['name'], 'raster', 'cfg.json')
    with open(cfg_fn, 'w') as f: json.dump(cfg, f)
    print('cfg saved!')





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