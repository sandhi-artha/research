import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

import rasterio as rs
from rasterio import plot
from rasterio import features as feat  # convert to mask

import geopandas as gpd

import tensorflow as tf

from lib.raster import read_slc, fix_orient
from lib.tiling import get_region_index, get_tile_index, tile_image_mask




# TFRecord data type
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor. intended for the image data
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_mask(mask, resize):
    mask = mask.astype(np.uint8)
    mask = np.expand_dims(mask, axis=-1)
    return tf.io.encode_png(mask)

def encode_image(image, resize):
    image = tf.image.resize(image, (resize,resize), method='nearest')
    return tf.io.encode_png(image)


# USER DATA
filt = 2
resize = 640
print(f'using multi-look filter: {filt}, and resize to: {resize}x{resize}')

# INITIALIZE
# get: processed_slc, original_slc and labels

# INPUT
print('reading acquisition dates')
with open('acq_dates.txt') as f:
    acq_list = f.readlines()
# cleaning
acq_list = [acq.split('.')[0] for acq in acq_list]
print(f'found total: {len(acq_list)}')

# try 1 first
acq_list = [acq_list[0]]
# acq_list = ['20190823162315_20190823162606']

pr_slc_dir = '../../processed'
or_slc_dir = '../../expanded-dataset'

print('reading labels..')
gt20_tr_gdf = gpd.read_file('../../expanded/geojson_buildings/SN6_AOI_11_Rotterdam_Buildings_GT20sqm-Train.geojson')

# OUTPUT
out_dir = f'../../tfrec{resize}_f{filt}'





for na, acq in enumerate(acq_list):
    print(f'preparing: stripe {na} out of {len(acq_list)}..')
    # 1. prepare the 3 ingredients
    pr_slc_path = f'{pr_slc_dir}/SLC_POL_{acq}_f{filt}.tif'
    pr_slc = read_slc(pr_slc_path)

    or_slc_path = f'{or_slc_dir}/CAPELLA_ARL_SM_SLC_HH_{acq}.tif'
    or_slc = rs.open(or_slc_path)

    cr_label = feat.geometry_mask(
        gt20_tr_gdf.geometry,
        out_shape=(or_slc.height, or_slc.width),
        transform=or_slc.transform,
        invert=True)

    pr_slc = fix_orient(pr_slc)
    cr_label = fix_orient(cr_label)

    # 2. prepare for tiling
    bound_idx = get_region_index(cr_label)
    # print(bound_idx)
    tile_list = get_tile_index(1280, 320, bound_idx)
    print(f'total tiles: {len(tile_list)}')
    
    # 3. start tiling
    # name is 20190823162315_20190823162606-75.tfrec
    if not os.path.exists(out_dir):
        print(f'creating {out_dir} directory..')
        os.makedirs(out_dir)

    tf_rec_fn = f'{out_dir}/{acq}-{len(tile_list)}.tfrec'
    print(f'Writing tfrec of {len(tile_list)} tiles..')
    with tf.io.TFRecordWriter(tf_rec_fn) as writer:
        for nt, tile_idx in enumerate(tile_list):
            # grab the data
            ex_img, ex_mask = tile_image_mask(pr_slc, [0,3,2], cr_label,
                                              tile_idx, bound_idx)
            
            # process image and mask
            ex_img = encode_image(ex_img, resize)
            ex_mask = encode_mask(ex_mask, resize)

            # create tfrecord structure
            feature = {'image': _bytes_feature(ex_img),
                       'mask': _bytes_feature(ex_mask)}
            
            # write tfrecords
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
            # report each 50th image
            if nt%50==0:
                print(nt)

    # save as image instead of tfrec
    # for nt, tile_idx in enumerate(tile_list):
    #     ex_img, ex_mask = tile_image_mask(pr_slc, [0,3,2], cr_label,
    #                                     tile_idx, bound_idx)

    #     # save image
    #     ex_img = Image.fromarray(ex_img)
    #     ex_img.save(f'{out_dir}/{acq}_{filt}_{nt:05}.png')

    #     fn_mask = f'{out_dir}/{acq}_{filt}_{nt:05}_mask.png'
    #     if not(os.path.isfile(fn_mask)):  # check if file doesn't exist, create it
    #         ex_mask = Image.fromarray(ex_mask)
    #         ex_mask.save(fn_mask)
        