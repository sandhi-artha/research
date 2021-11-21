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
import pickle

from dataset_cfg import cfg
from sar_preproc import SarPreproc
from tile_gen import get_label_gdf, raster_vector_tiling

from shapely.geometry import box
import lib.tfrec as tfrec

# load all timestamps and their sar orientation
with open('timestamp_orientation.pickle','rb') as f:
    time_orient = pickle.load(f)

# split dataset based on each label split
print('loading labels')
labels = {
    'train' : get_label_gdf('train', cfg["label_dir"]),
    'val'   : get_label_gdf('val', cfg["label_dir"]),
    'test'  : get_label_gdf('test', cfg["label_dir"])
}
bounds = {
    'train' : box(*labels["train"].total_bounds),
    'val'   : box(*labels["val"].total_bounds),
    'test'  : box(*labels["test"].total_bounds)
}
print(f'labels loaded')

# loop through timestamps
# for to in time_orient[:2]:
#     print(f'processing raster for {to}')
#     timestamp = to[:-2]
#     orient = to[-1]

#     # process SAR
#     out_fn = 'output.tif'  # f'{to}.tif'  # give specific output
#     sar_preproc = SarPreproc(cfg, timestamp, cfg["in_dir"], cfg["out_dir"], out_fn)
#     sar_preproc()

#     print(f'creating tiles')
#     # tile raster and vector
#     tile_in_path = os.path.join(cfg["out_dir"], out_fn)  # output.tif path
#     tile_out_path = os.path.join(cfg["out_dir"], cfg["name"])
#     raster_dict, vector_dict = raster_vector_tiling(
#         cfg, labels, bounds, timestamp, tile_in_path, tile_out_path)
    
#     print('tiling complete')
#     break

# write tfrecords
in_path = tile_out_path = os.path.join(cfg["out_dir"], cfg["name"])
for split in ['train','val','test']:
    # grab and shuffle raster-vector file paths
    fp = tfrec.get_tile_paths(in_path, split, shuffle=True)
    
    # create save directory if not exist
    base_dir = os.path.join(cfg["out_dir"], 'tfrecord')
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    
    # name of tfrec file
    base_fn = f'tfrecord/{split}'
    base_fn = os.path.join(base_dir, split)
    print(base_fn)
    tfrec.create_tfrecord(*fp, cfg, base_fn)