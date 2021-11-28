cfg = {
    # sar-preproc
    'pol'           : ['HH', 'HV', 'VH', 'VV'],
    'ml_filter'     : 'avg',  # 'avg', 'med'
    'ml_size'       : 2, # 2, 4, 8
    'in_dir'        : '../../dataset/sn6-expanded', # '../../expanded-dataset', # where the SLC stripes located
    # tiling
    'project'       : 'sensor',
    'name'          : 'base',
    'stride'        : 0, # 0, 80 unit in px
    'out_dir'       : '../../dataset/sensor', # '../../sensor',  # where the tiles will be stored
    'label_dir'     : '../../dataset/spacenet6-challenge/expanded/exp_geojson_buildings', # '../../expanded/geojson_buildings',  
    'load_tile'     : 0,  # 1 means load from tile_scheme folder instead of generating from scratch
    'verbose'       : 0,  # 1 for all info, 2 for necessary tiling
    # post-tiling
    'tfrec_dir'     : 'tfrecord',   # folder to save tfrecords, change with post-tile versions
    'load_fix'      : 1,            # 1 means load from vector_fix instead of vector
    'channel'       : [1,4,3],
    'out_precision' : 8,            # 8, 16 or 32
    'tfrec_size'    : 100,          # num examples per tfrec
    'perc_data'     : 1.0,          # percentage of training data
    'rotation'      : 0,
    'flip'          : 0,
}

"""
folder structure
{out_dir}
    {name}
        raster
        {tfrec_dir}
    s0
        tile_scheme
        vector
        vector_fix
    s80
        tile_scheme
        vector
        vector_fix
    output.tif

"""