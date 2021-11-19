import solaris.raster_tile as raster_tile
import solaris.vector_tile as vector_tile
from shapely.geometry import box
import os
import geopandas as gpd

def get_label_gdf(split, in_dir):
    """split: str. 'train', 'val', 'test'
    in_dir: str. root path to folder containing .geojson
    """
    fn = f'SN6_AOI_11_Rotterdam_Buildings_GT20sqm-{split.capitalize()}.geojson'
    return gpd.read_file(os.path.join(in_dir, fn))

def raster_vector_tiling(cfg, labels, bounds, timestamp, in_path, out_path):
    """creates
    """
    raster_dict = {}
    vector_dict = {}
    
    for split in ['train','val','test']:
        fn = '{}_{}_{}_{}_s{}'.format(
            cfg["project"],
            timestamp,
            cfg["name"],
            split,
            cfg["stride"]
        )

        # tile the raster
        raster_tiler = raster_tile.RasterTiler(dest_dir=os.path.join(out_path,'raster'), 
                                       src_tile_size=(640, 640),
                                       aoi_boundary=bounds[split],
                                       verbose=True,
                                       stride=(cfg["stride"],cfg["stride"]))
        
        raster_tiler.tile(in_path, dest_fname_base=fn, nodata_threshold=0.5)

        # use created tiles for vector tiling
        vector_tiler = vector_tile.VectorTiler(dest_dir=os.path.join(out_path,'vector'),
                                               super_verbose=True)
        
        vector_tiler.tile(labels[split], tile_bounds=raster_tiler.tile_bounds,
                          split_multi_geoms=False, dest_fname_base=fn)
        
        raster_dict[split] = raster_tiler
        vector_dict[split] = vector_tiler
    
    # return for debugging
    return raster_dict, vector_dict