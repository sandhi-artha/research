import os
import glob
import numpy as np
import random
import geopandas as gpd
import tensorflow as tf
import rasterio as rs
from rasterio import features as feat

from solaris.vector_mask import mask_to_poly_geojson
from solaris.core import save_empty_geojson
from .proc import hist_clip, to_hwc, normalize


def get_tile_paths(cfg, split, shuffle=False):
    """
    in_path : str
        path where tiles are stored (has [raster, vector])
    split : str
        'train', 'val', or 'test'
    shuffle : bool
        shuffle using a seed
    perc_data : float [0,1]
        only applies to train split, percentage of examples to be loaded
    returns : [raster_paths, vector_paths]
        list of all raster paths and vector paths for given split
    """
    path = os.path.join(cfg["out_dir"], cfg["name"], 'raster', f'*{split}*.tif')
    raster_paths = glob.glob(path)
    raster_paths.sort()

    if shuffle:
        random.Random(17).shuffle(raster_paths)
    if split=='train':
        tot_len = len(raster_paths)
        per_len = int(np.ceil(tot_len*cfg["perc_data"]))
        raster_paths = raster_paths[:per_len]

    vect_str = 'vector_fix' if cfg["load_fix"] else 'vector'
    vector_paths = []
    for rp in raster_paths:
        vp = rp.replace('raster', vect_str)
        vp = vp.replace('tif','geojson')
        vp = vp.replace(cfg["name"], f's{cfg["stride"]}')
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
        for float32, nan will be replaced by 0.0
        for uint, casting auto converts nan to 0.0
    """
    if out_precision==8:
        dtype = tf.uint8
        image = tf.cast(image*(2**8 - 1), dtype=dtype)
    elif out_precision==16:
        dtype = tf.uint16
        image = tf.cast(image*(2**16 - 1), dtype=dtype)
    else:
        dtype = tf.float32
        image = tf.cast(np.nan_to_num(image, nan=0.0), dtype=dtype)

    image_tensor = tf.constant(image, dtype=dtype)
    image_serial = tf.io.serialize_tensor(image_tensor)
    return image_serial


### PROC LABEL ###
def clip_vector_mask(raster_path, vector_path):
    """reads the mask from raster, and clips the vector using it
        when input vector is empty, will return an empty gdf
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

    # read vector and clip it
    vector = gpd.read_file(vector_path)
    vector_fix = vector.clip(mask_gdf)

    save_fn = vector_path.replace('vector', 'vector_fix')
    if vector_fix.shape[0] == 0:
        save_empty_geojson(save_fn, crs=raster.crs)
    else:
        vector_fix.to_file(save_fn, driver='GeoJSON')

    raster.close()  # close the opened dataset
    return vector_fix

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

def get_label(raster_path, vector_path, load_fix=0):
    """
    load_fix : bool
        if True, loads from existing "vector_fix" folder instead
        of creating from scratch (must create vector_fix folder
        first, otherwise will throw error)
    returns : [bin_mask, vector]
        bin_mask is type bool, vector is type geodataframe
    """
    if load_fix:
        vector_fix = gpd.read_file(vector_path)
    else:
        vector_fix = clip_vector_mask(raster_path, vector_path)

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
def create_tfrecord(raster_paths, vector_paths, cfg, base_fn, size):
    """
    100 images per tfrecord
    image in float32 serialized
    mask in binary serialized
    size : int
        examples per tfrecord. for 3ch, 640 res, uint8, use 100 -> ~150MB/file
    output : {base_fn}01-100.tfrec
    """
    tot_ex = len(raster_paths)  # total examples
    tot_tf = int(np.ceil(tot_ex/size))  # total tfrecords

    for i in range(tot_tf):
        print(f'Writing TFRecord {i} of {tot_tf}..')
        size2 = min(size, tot_ex - i*size)  # size=size2 unless for remaining in last file
        fn = f'{base_fn}{i:02}-{size2}.tfrec'

        with tf.io.TFRecordWriter(fn) as writer:
            for j in range(size2):
                idx = i*size+j  # ith tfrec * num_img per tfrec as the start of this iteration
                image = get_image(raster_paths[idx], cfg['channel'])
                image_serial = serialize_image(image, cfg['out_precision'])

                label, label_gdf = get_label(raster_paths[idx], vector_paths[idx])
                label_serial = serialize_label(label)

                fn = os.path.basename(raster_paths[idx]).split('.')[0]

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

def create_tfrecord_parallel(proc_idx, base_fn, split, cfg):
    
    """divides task between process based on num of tfrecords of a given split
    """
    raster_paths, vector_paths = get_tile_paths(cfg, split, shuffle=True)
    size = cfg["tfrec_size"]
    tot_ex = len(raster_paths)  # total examples
    tot_tf = int(np.ceil(tot_ex/size))  # total tfrecords
    
    # proc_idx is same as i in create_tfrecord
    print(f'Writing TFRecord {proc_idx} of {tot_tf}..')
    size2 = min(size, tot_ex - proc_idx*size)  # size=size2 unless for remaining in last file
    fn = f'{base_fn}{proc_idx:02}-{size2}.tfrec'
    with tf.io.TFRecordWriter(fn) as writer:
        for j in range(size2):
            idx = proc_idx*size+j  # ith tfrec * num_img per tfrec as the start of this iteration
            image = get_image(raster_paths[idx], cfg["channel"])
            image_serial = serialize_image(image, cfg["out_precision"])

            label, label_gdf = get_label(raster_paths[idx], vector_paths[idx], cfg["load_fix"])
            label_serial = serialize_label(label)

            fn = os.path.basename(raster_paths[idx]).split('.')[0]

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
        pass


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