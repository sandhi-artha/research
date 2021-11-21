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
    'in_dir'        : '../../dataset/sn6-expanded',  # where the SLC stripes located
    'out_dir'       : '../../dataset/sensor',  # where the tiles will be stored
    'label_dir'     : '../../dataset/spacenet6-challenge/expanded/exp_geojson_buildings',
    'verbose'       : 0,
    # post-tiling
    'channel'       : [1,4,3],
    'out_precision' : 32,  # 8, 16 or 32
    'clip_thresh'   : 80, # num of pixels to clip
}