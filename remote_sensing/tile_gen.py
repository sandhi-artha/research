import os
import pickle

import rasterio as rs
import geopandas as gpd
from rasterio import windows
from shapely.geometry import box

import solaris.raster_tile as raster_tile
import solaris.vector_tile as vector_tile

def get_label_gdf(split, in_dir):
    """split: str. 'train', 'val', 'test'
    in_dir: str. root path to folder containing .geojson
    """
    fn = f'SN6_AOI_11_Rotterdam_Buildings_GT20sqm-{split.capitalize()}.geojson'
    return gpd.read_file(os.path.join(in_dir, fn))

def get_labels_bounds(label_dir):
    """returns [labels, bounds]
    each is dictionary containing 'train','val','test' split
    labels : geodataframe of building footprints
    bounds : shapely. boundary/extent where labels are available
    """
    labels = {
        'train' : get_label_gdf('train', label_dir),
        'val'   : get_label_gdf('val', label_dir),
        'test'  : get_label_gdf('test', label_dir)
    }

    # give paddings to ensure same raster and vector coverage
    # when using total_bounds, the train boundaries take square coverage
    #   so raster at bot right corner has image but with labels cutoff
    #   +- 100 for val is not enough (blame the tiling algo)
    train_bounds = labels['train'].total_bounds
    train_bounds[2] -= 400

    val_bounds = labels['val'].total_bounds
    val_bounds[0] += 400
    val_bounds[2] -= 400

    test_bounds = labels['test'].total_bounds
    test_bounds[0] += 400
    test_bounds[2] -= 400

    bounds = {
        'train' : box(*train_bounds),
        'val'   : box(*val_bounds),
        'test'  : box(*test_bounds)
    }

    return labels, bounds

def save_tile_scheme(cfg, timestamp, orient, split, tiler):
    """tile_schemes are unique depending on stride
    a pickle is save for every timestamp and each of its split, contains:
        [tile_names, tile_bounds, tile_transform]

    this {timestamp}_s{stride}.pickle can be used to quickly create tiling
    """
    out_path = os.path.join(cfg['out_dir'], f"s{cfg['stride']}", 'tile_scheme')
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    fn = '{}_{}_{}.pickle'.format(timestamp, orient, split)
    fn_path = os.path.join(out_path, fn)
    with open(fn_path, 'wb') as f:
        pickle.dump(tiler.tile_scheme, f)


def load_scheme(cfg, timestamp, orient, split):
    """scheme contains [name, bound, profile] for each filtered tiles (above nodata_threshold)
    """
    out_path = os.path.join(cfg['out_dir'], f"s{cfg['stride']}", 'tile_scheme')
    fn = '{}_{}_{}.pickle'.format(timestamp, orient, split)
    fn_path = os.path.join(out_path, fn)
    with open(fn_path, 'rb') as f:
        scheme = pickle.load(f)
    return scheme

def parallel_tile_generator(scheme, slc_in, out_dir):
    """used boundaries from scheme for tiling raster only
    """
    # used up to 350MB ram each proc
    src = rs.open(slc_in)
    dest_fname,tb,profile = scheme
    # get window using tile resolution
    window = windows.from_bounds(
        *tb, transform=src.transform,
        width=640,
        height=640)

    # crop main raster using window
    tile_data = src.read(window=window, boundless=True, fill_value=src.nodata)

    # save tile_raster
    fn_path = os.path.join(out_dir, 'raster', dest_fname)
    with rs.open(fn_path, 'w', **profile) as dest:
        for band in range(1, profile['count'] + 1):
            dest.write(tile_data[band-1, :, :], band)
        dest.close()

    src.close()


def raster_vector_tiling(cfg, timestamp, orient, slc_path, out_dir):
    """create raster and vector tiles using Tiler from solaris
    """
    labels, bounds = get_labels_bounds(cfg["label_dir"])

    raster_dict = {}
    vector_dict = {}
    vector_save_path = os.path.join(cfg['out_dir'], f"s{cfg['stride']}", 'vector')
    
    for split in ['train','val','test']:
        fn = '{}_{}_o{}_{}_{}_s{}'.format(
            cfg["project"],
            timestamp,
            orient,
            cfg["name"],
            split,
            cfg["stride"]
        )

        # tile the raster
        raster_tiler = raster_tile.RasterTiler(dest_dir=os.path.join(out_dir,'raster'), 
                                       src_tile_size=(640, 640),
                                       aoi_boundary=bounds[split],
                                       verbose=cfg["verbose"],
                                       stride=(cfg["stride"],cfg["stride"]))
        
        raster_tiler.tile(slc_path, dest_fname_base=fn, nodata_threshold=0.5)
        save_tile_scheme(cfg, timestamp, orient, split, raster_tiler)

        # use created tiles for vector tiling
        vector_tiler = vector_tile.VectorTiler(dest_dir=vector_save_path,
                                               super_verbose=cfg["verbose"])
        
        vector_tiler.tile(labels[split], tile_bounds=raster_tiler.tile_bounds,
                          split_multi_geoms=False, dest_fname_base=fn)
        
        raster_dict[split] = raster_tiler
        vector_dict[split] = vector_tiler
        
    # return for debugging
    return raster_dict, vector_dict