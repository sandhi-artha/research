cfg = {
    'pol'           : ['HH', 'HV', 'VH', 'VV'],
    'ml_filter'     : 'avg',  # 'avg', 'med'
    'ml_size'       : 2, # 2, 4, 8
    'coverage'      : 1, # 1, 0.5, 0.25
    'stride'        : 0, # 0, 80 unit in px
    'project'       : 'sensor',
    'name'          : 'base',
    'rotation'      : 0,
    'flip'          : 0,
    'in_dir'        : '../../dataset/sn6-expanded', # '../../expanded-dataset', # where the SLC stripes located
    'out_dir'       : '../../dataset/sensor', # '../../sensor',  # where the tiles will be stored
    'label_dir'     : '../../dataset/spacenet6-challenge/expanded/exp_geojson_buildings', # '../../expanded/geojson_buildings',  
    'tfrec_dir'     : 'tfrecord',  # folder to save tfrecords, change with post-tile versions
    'verbose'       : 0,  # 1 for all info, 2 for necessary tiling
    # post-tiling
    'channel'       : [1,4,3],
    'out_precision' : 8,  # 8, 16 or 32
    'tfrec_size'    : 100, # num examples per tfrec
    'perc_data'     : 1.0,  # percentage of training data
}