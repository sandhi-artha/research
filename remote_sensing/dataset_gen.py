# input: SN6-expanded dataset
# preprocess SAR according to dataset-scheme
# saves to temporary raster 'output.tif' (to minimize storage usage)
# read saved raster and do tiling according to regions
# output tile name: sensor_20190823162315_20190823162606_base_train_100_0016.tif
# project_ timestamp_ dsscheme_ split_ coverage_ tile_id


# import matplotlib.pyplot as plt
# import numpy as np
# import os

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

in_dir = '../../expanded-dataset'
out_dir = f'../../sensor'


with open('timestamp_orientation.pickle','rb') as f:
    time_orient = pickle.load(f)

for to in time_orient[:3]:
    print(f'processing: {to}')
    timestamp = to[:-2]
    orient = to[-1]
    out_fn = f'{to}.tif'
    sar_preproc = SarPreproc(cfg, timestamp, in_dir, out_dir, out_fn)
    sar_preproc()