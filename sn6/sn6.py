import pandas as pd
import numpy as np
import tensorflow as tf

import geopandas as gpd     # geospatial libraries
import rasterio as rs
from rasterio import features as feat
from rasterio.plot import show

from shapely.geometry import Point, Polygon     # polygon creation
import matplotlib.pyplot as plt     # plotting
import glob # file handling
import os
import cv2  # filter response


from .processing import norm, get_region_index, mask2contour
from .filtering import filter_image



# TFRecord data type
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor. intended for the image data
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))




class Sn6Create():
    def __init__(self, ds_path,
                 split_path=None,
                 mode='PS-RGB',
                 image_size=(900,900),
                 sar_ch=[1,2,3,4],
                 filter=None,
                 r=0,
                 crop=False
                ):
        """Class for preprocessing and createing TFRecord
        Parameters
        ----------
        ds_path : str
            folder path where it ends with 'AOI_11_Rotterdam/' don't forget the slash
        
        split_path : str
            path to the .npy file containing tile names of dataset split. required for
            tfrecords creation
        
        mode : str, default: 'PS-RGB'
            sub folder of what image type to use, correct values: 
            'PAN', 'PS-RGB', 'PS-RGBNIR', 'SAR-Intensity'
        
        image_size : (w,h), default: (900,900)
            tuple, resolution for tfrecords output
            
        sar_ch : list, default: [1,2,3,4]
            choose which sar channels will be used, [1,2,3,4] for [HH,VH,HV,VV]
            also decides order. If not provided, will use all 4ch in default order
        
        filter : str, default: None
            filter the image or not, correct values:
            'average', 'hamming'

        r : int, default: 0
            filter strength

        crop: bool, default: False
            if True, 1 tile will result in 2 images where the NoData region (black parts)
            are cropped. Cropping forces resolution output to (512,512)
        """
        self.path = ds_path
        self.mode = mode
        self.sar_ch = sar_ch
        self.filter = filter
        self.r = r
        self.crop = crop
        
        if crop:
            self.image_size = (512,1600)    # when crop enabled, max output size is 512
        else:
            self.image_size = image_size
        
        # load the split set
        if split_path:
            self.split_set = np.load(split_path, allow_pickle=True)


    def get_image_path(self, image_id):
        return f'{self.path}{self.mode}/SN6_Train_AOI_11_Rotterdam_{self.mode}_{image_id}.tif'

    def get_binary_mask(self, image_id, raster):
        """Create mask where 1 are pixel values of buildings and 0 for others
        polygons are transformed using raster's transformation
        returns: binary_mask, np.array of 0 and 1
        """
        # get geojson file for a given tile
        geo_path = f'{self.path}geojson_buildings/SN6_Train_AOI_11_Rotterdam_Buildings_{image_id}.geojson'
        gdf = gpd.read_file(geo_path)

        # handle error when no buildings are present in a tile
        if gdf.shape[0]==0:
            if self.crop:
                mask = np.zeros((900,900))
            else:
                mask = np.zeros(self.image_size)
        else:
            # create binary mask, convert to uint8, resize
            mask = feat.geometry_mask(
                gdf.geometry,
                out_shape=(raster.height, raster.width), # original wxh (900,900)
                transform=raster.transform,
                invert=True  # makes pixel buildings == 1
            )

        return mask, gdf.shape[0]

    def get_mask_string(self, mask, image_id=None):
        """
        encodes to string for tfrecord creation
        if crop=True, returns 2 masks, if false, returns mask in mask1
        """
        # encode mask to png
        mask = mask.astype(np.uint8)
        mask = np.expand_dims(mask, axis=2)  # result: (w,h,1)
        mask = tf.image.resize(mask, size=self.image_size, method='nearest', preserve_aspect_ratio=True)
        if self.crop:
            mask1 = mask[:,:512,:]
            mask2 = mask[:,-512:,:]
            
            assert mask1.shape == (512,512,1), f'{image_id}, mask1 is {mask1.shape}'
            assert mask2.shape == (512,512,1), f'{image_id}, mask2 is {mask2.shape}'
            
            mask1 = tf.io.encode_png(mask[:,:512,:])
            mask2 = tf.io.encode_png(mask[:,-512:,:])
            return [mask1,mask2]
        else:
            return [tf.io.encode_png(mask)]
    

    def get_image_string(self, image):
        """process the image raster
        hierarchy of options:
            crop or not
                sar or not
                filter or not

        """
        image = rs.plot.reshape_as_image(image)  # fix dimension order, res (w,h,ch)

        if self.crop:
            if self.mode == 'SAR-Intensity':
                image = tf.image.resize(image, size=self.image_size, preserve_aspect_ratio=True)
            else:
                image = tf.image.resize(image, size=self.image_size, preserve_aspect_ratio=True, method='nearest')
            
            # crop and get 2 square images as numpy array
            image1 = image[:,:512,:].numpy()  # from start take 512x512
            image2 = image[:,-512:,:].numpy()  # from last take 512x512

            if self.r > 0:
                # apply filter to each dimension separately
                for i in range(image.shape[-1]):
                    image1[:,:,i] = filter_image(self.filter, image1[:,:,i], self.r)
                    image2[:,:,i] = filter_image(self.filter, image2[:,:,i], self.r)

            # encode image, change arr to uint8
            image1 = tf.io.encode_png(image1.astype(np.uint8))
            image2 = tf.io.encode_png(image2.astype(np.uint8))

            return [image1, image2]

        else:
            if self.mode == 'SAR-Intensity':
                image = norm(image)
            
            if self.r > 0:
                image = filter_image(self.filter, image, self.r)

            image = tf.io.encode_png(image)

            return [image]
        


    def get_image_mask(self, image_id):
        """Takes an image id, ex: '20190822065725_20190822065959_tile_7283'

        returns:
            image (arr str), mask (arr string)

        """
        # read image with rasterio
        raster = rs.open(self.get_image_path(image_id))

        mask, num_building = self.get_binary_mask(image_id, raster)

        if self.crop:
            row0,row1,col0,col1 = get_region_index(raster)  # windowing

            # prepare mask ROI
            mask = mask[row0:row1,col0:col1]

            # prepare image ROI
            if self.mode == 'SAR-Intensity':
                image = raster.read(indexes=self.sar_ch, window=((row0,row1),(col0,col1)))
            else:
                image = raster.read(out_dtype='uint8', window=((row0,row1),(col0,col1)))
        else:
            # prepare image
            if self.mode == 'SAR-Intensity':
                image = raster.read(indexes=self.sar_ch, out_shape=self.image_size)
            else:
                image = raster.read(out_dtype='uint8', out_shape=self.image_size)

        
        masks = self.get_mask_string(mask, image_id)        # list of mask (1 if crop=False, 2 if True)
        images = self.get_image_string(image)

        return images, masks #, num_building




    def create_tfrecord_crop(self):
        print(f'using {self.image_size[0]}x{self.image_size[0]} resolution on {self.mode} images')

        ni = 1 if self.crop else 2  # number of images per tile

        # create tfrecords for each split
        for n, image_set in enumerate(self.split_set):
            print(f'writing split {n+1} of {len(self.split_set)}')
            fn = f'split{n+1}-{ni*len(image_set)}.tfrec'

            with tf.io.TFRecordWriter(fn) as writer:
                for k,image_id in enumerate(image_set):
                    image_str, mask_str = self.get_image_mask(image_id)
                    
                    for i in range(ni):
                        feature={
                            'image': _bytes_feature(image_str[i]),
                            'mask': _bytes_feature(mask_str[i]),
                            'file_name': _bytes_feature(tf.compat.as_bytes(f'{image_id}_{i}'))
                        }
                        # write tfrecords
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())
                    
                    if k%50==0:
                        print(f'split {n+1}: {k}')