import os
import glob

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rs

from lib.viz import *
from lib.tfrec import get_vector_bin

def get_tile_stats(raster_path, vector_path):
    raster = rs.open(raster_path)
    raster_img = raster.read([1], masked=True)
    tot_pixels = raster_img.shape[1] * raster_img.shape[2]
    nodata_pixels = tot_pixels - raster_img.count()
    perc_nodata = nodata_pixels/tot_pixels*100

    vector = gpd.read_file(vector_path)
    mask = get_vector_bin(raster_path, vector)
    building_pixels = mask.sum()
    perc_building = building_pixels/tot_pixels*100

    raster.close()
    return perc_nodata, perc_building

if __name__=='__main__':
    raster_dir = '../../dataset/sensor/base/raster'
    vector_dir = '../../dataset/sensor/s0/vector_fix'

    raster_paths = glob.glob(os.path.join(raster_dir, '*train*.tif'))
    raster_paths.sort()
    vector_paths = glob.glob(os.path.join(vector_dir, '*train*.geojson'))
    vector_paths.sort()
    
    perc_nodatas = []
    perc_buildings = []
    tilenames = []
    for i in range(len(raster_paths)):
        perc_nodata, perc_building = get_tile_stats(raster_paths[i], vector_paths[i])
        perc_nodatas.append(perc_nodata)
        perc_buildings.append(perc_building)
        tilenames.append(os.path.basename(raster_paths[i]))

    df = pd.DataFrame({
        'raster': raster_paths,
        'perc_nodata': perc_nodatas,
        'perc_buildings': perc_buildings
    })
    
    df.to_csv('tile_stats.csv', index=False)