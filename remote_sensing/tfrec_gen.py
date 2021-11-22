import time
import os
from dataset_cfg import cfg
import lib.tfrec as tfrec


start = time.time()
in_path = tile_out_path = os.path.join(cfg["out_dir"], cfg["name"])
for split in ['train','val','test']:
    # grab and shuffle raster-vector file paths
    fp = tfrec.get_tile_paths(in_path, split, shuffle=True, perc_data=cfg["perc_data"])
    
    # create save directory if not exist
    base_dir = os.path.join(cfg["out_dir"], cfg["name"], 'tfrecord8')
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    
    # name of tfrec file
    base_fn = os.path.join(base_dir, split)
    tfrec.create_tfrecord(*fp, cfg, base_fn)

end = time.time()
print(f'tfrecords created in {(end-start):.1f}s')