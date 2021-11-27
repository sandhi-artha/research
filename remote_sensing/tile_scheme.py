"""
each time_stamp has a pickle object containing:
- tile_names
- tile_bounds
- tile_transform

this {timestamp}_s{stride}.pickle can be used to quickly create tiling
"""
import os
import pickle
import time
from rasterio import windows
import rasterio as rs

def save_raster_vector_tiling(cfg, timestamp, orient):

    pass

def load_scheme(out_path, timestamp, orient, split):
    """scheme contains [name, bound, profile] for each filtered tiles (above nodata_threshold)
    """
    fn = '{}_{}_{}.pickle'.format(timestamp, orient, split)
    fn_path = os.path.join(out_path, fn)
    with open(fn_path, 'rb') as f:
        scheme = pickle.load(f)
    return scheme

def simple_tile_generator(in_raster_path, out_path, scheme, src_tile_size):
    """snippet from raster_tile.tile_generator
    made for specific settings
    
    scheme:
        [[name, bound, profile],...]
    """
    src = rs.open(in_raster_path)
    for dest_fname,tb,profile in scheme:
        # get window using tile resolution
        window = windows.from_bounds(
            *tb, transform=src.transform,
            width=src_tile_size[1],
            height=src_tile_size[0])

        # crop main raster using window
        tile_data = src.read(window=window, boundless=True, fill_value=src.nodata)

        dest_path = os.path.join(out_path, 'raster_loaded', dest_fname)
        # save tile_raster
        with rs.open(dest_path, 'w', **profile) as dest:
            for band in range(1, profile['count'] + 1):
                dest.write(tile_data[band-1, :, :], band)
            dest.close()

    src.close()


def load_raster_vector_tiling(timestamp, orient, in_path, out_path):
    """
    """
    for split in ['train','val','test']:
        start = time.time()
        scheme = load_scheme(out_path, timestamp, orient, split)
        simple_tile_generator(in_path, out_path, scheme, (640,640))
        
        end = time.time()
        print(f'finished {split} split in {(end-start):.1f}s')
