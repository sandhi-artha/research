import solaris.raster_tile as raster_tile
import solaris.vector_tile as vector_tile
from shapely.geometry import box
import os
import geopandas as gpd
import time
import pickle

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
    a pickle is save for every timestamp and each of its split
    """
    out_path = os.path.join(cfg['out_dir'], f"s{cfg['stride']}", 'tile_scheme')
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    fn = '{}_{}_{}.pickle'.format(timestamp, orient, split)
    fn_path = os.path.join(out_path, fn)
    with open(fn_path, 'wb') as f:
        pickle.dump(tiler.tile_scheme, f)

def raster_vector_tiling(cfg, labels, bounds, timestamp, orient, in_path, out_path):
    """creates
    """
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

        start = time.time()
        # tile the raster
        raster_tiler = raster_tile.RasterTiler(dest_dir=os.path.join(out_path,'raster'), 
                                       src_tile_size=(640, 640),
                                       aoi_boundary=bounds[split],
                                       verbose=cfg["verbose"],
                                       stride=(cfg["stride"],cfg["stride"]))
        
        raster_tiler.tile(in_path, dest_fname_base=fn, nodata_threshold=0.5)
        print('saving scheme')
        save_tile_scheme(cfg, timestamp, orient, split, raster_tiler)

        # use created tiles for vector tiling
        vector_tiler = vector_tile.VectorTiler(dest_dir=vector_save_path,
                                               super_verbose=cfg["verbose"])
        
        vector_tiler.tile(labels[split], tile_bounds=raster_tiler.tile_bounds,
                          split_multi_geoms=False, dest_fname_base=fn)
        
        raster_dict[split] = raster_tiler
        vector_dict[split] = vector_tiler
        
        end = time.time()
        print(f'finished {split} split in {(end-start):.1f}s')
    
    # return for debugging
    return raster_dict, vector_dict