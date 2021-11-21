import os
import glob
import numpy as np
import random
import geopandas as gpd
import tensorflow as tf
import rasterio as rs
from rasterio import features as feat

from solaris.vector_mask import mask_to_poly_geojson
from .proc import hist_clip, to_hwc, normalize


def get_tile_paths(in_path, split, shuffle=False):
    """
    in_path : str
        path where tiles are stored (has [raster, vector])
    split : str
        'train', 'val', or 'test'
    shuffle : bool
        shuffle using a seed
    returns : [raster_paths, vector_paths]
        list of all raster paths and vector paths for given split
    """
    path = os.path.join(in_path,'raster',f'*{split}*.tif')
    raster_paths = glob.glob(path)
    raster_paths.sort()

    if shuffle:
        random.Random(17).shuffle(raster_paths)

    vector_paths = []
    for rp in raster_paths:
        vp = rp.replace('raster', 'vector')
        vp = vp.replace('tif','geojson')
        vector_paths.append(vp)

    return raster_paths, vector_paths


### PROC IMAGE ###
def get_image(raster_path, ch=None, thresh=80):
    """
    ch = list or int
        starts at 1, if None, return all channels
    returns: np.array
        type same as raster (float32), range [0,1]
    """
    raster = rs.open(raster_path)
    image = raster.read(indexes=ch, masked=True)
    image = hist_clip(image, thresh=thresh)
    image = to_hwc(image)
    image = normalize(image)
    raster.close()  # close the opened dataset
    return image

def serialize_image(image, out_precision=32):
    """
    image : np.array
        image with pixel range 0-1
    out_precision : int
        8, 16 or 32 for np.uint8, np.uint16 or np.float32
    """
    if out_precision==8:
        dtype = tf.uint8
        image = tf.cast(image*(2**8 - 1), dtype=dtype)
    elif out_precision==16:
        dtype = tf.uint16
        image = tf.cast(image*(2**16 - 1), dtype=dtype)
    else:
        dtype = tf.float32

    image_tensor = tf.constant(image, dtype=dtype)
    image_serial = tf.io.serialize_tensor(image_tensor)
    return image_serial


### PROC LABEL ###
def clip_vector_mask(raster_path, vector):
    """reads the mask from raster, and clips the vector using it
    raster_path : str
        path to .tif raster that have mask
    vector : geopandas gdf
    """
    raster = rs.open(raster_path)
    mask = raster.read_masks([1])
    mask_gdf = mask_to_poly_geojson(
        mask[0], reference_im=raster_path,
        do_transform=True
    )

    raster.close()  # close the opened dataset

    # clip vector with mask gdf and return
    return vector.clip(mask_gdf)

def get_vector_bin(raster_path, vector):
    """
    returns : np.array type: bool
        mask array where pixel buildings=1 and background=0
    """
    raster = rs.open(raster_path)
    h = raster.height  # rows
    w = raster.width   # cols
    transform = raster.transform
    
    raster.close()  # close the opened dataset

    # handle when no buildings are in the tile
    if vector.shape[0]==0:
        mask = np.zeros((h,w),dtype=bool)
    else:
        mask = feat.geometry_mask(
            vector.geometry,
            out_shape=(h,w),
            transform=transform,
            invert=True  # pixel buildings == 1
        )

    return mask

def get_label(raster_path, vector_path):
    """
    returns : [bin_mask, vector]
        bin_mask is type bool, vector is type geodataframe
    """
    vector = gpd.read_file(vector_path)
    vector_fix = clip_vector_mask(raster_path, vector)
    bin_mask = get_vector_bin(raster_path, vector_fix)
    return bin_mask, vector_fix

def serialize_label(label):
    """
    label : np.array
        binary mask
    """
    label_tensor = tf.constant(label, dtype=tf.bool)
    label_serial = tf.io.serialize_tensor(label_tensor)
    return label_serial



### TFRecord DATA TYPE ###
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor. intended for the image data
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



### CREATE AND READ ###
def create_tfrecord(raster_paths, vector_paths, cfg, base_fn):
    """
    100 images per tfrecord
    image in float32 serialized
    mask in binary serialized
    output : {base_fn}01-100.tfrec
    """
    size = 100  # examples per tfrecord
    tot_ex = len(raster_paths)  # total examples
    tot_tf = int(np.ceil(tot_ex/size))  # total tfrecords

    for i in range(tot_tf):
        print(f'Writing TFRecord {i} of {tot_tf}..')
        size2 = min(size, tot_ex - i*size)  # size=size2 unless for remaining in last file
        fn = f'{base_fn}{i:02}-{size2}.tfrec'

        with tf.io.TFRecordWriter(fn) as writer:
            for j in range(size2):
                image = get_image(raster_paths[j], cfg['channel'], cfg['clip_thresh'])
                image_serial = serialize_image(image, cfg['out_precision'])

                label, label_gdf = get_label(raster_paths[j], vector_paths[j])
                label_serial = serialize_label(label)

                fn = os.path.basename(raster_paths[j]).split('.')[0]

                feature = {
                    'image': _bytes_feature(image_serial.numpy()),
                    'label': _bytes_feature(label_serial.numpy()),
                    'fn' : _bytes_feature(tf.compat.as_bytes(fn))
                }

                # write tfrecords
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
                
                if j%50==0:
                    print(f'{j} / {size2}')

def read_tfrecord(serialized_example):
    """
    assumes precision is tf.float32, change to tf.uint16 or tf.uint8 if required
    """
    tfrec_format = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'fn': tf.io.FixedLenFeature([], tf.string)
    }

    res_features = tf.io.parse_single_example(serialized_example, tfrec_format)
    image = tf.io.parse_tensor(res_features['image'], tf.float32)
    label = tf.io.parse_tensor(res_features['label'], tf.bool)
    fn = res_features['fn']
    return image, label, fn