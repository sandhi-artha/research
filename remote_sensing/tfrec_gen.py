import os
import json
import glob
from multiprocessing import Pool

from numpy import ceil

from timebudget import timebudget
from dataset_cfg import cfg
from lib.tfrec import create_tfrecord_parallel, get_tile_paths




def get_tot_tf(cfg, split):
    path = os.path.join(cfg["out_dir"], cfg["name"], 'raster', f'*{split}*.tif')
    tot_ex = len(glob.glob(path))    # total examples
    if split=='train':
        per_ex = int(ceil(tot_ex*cfg["perc_data"]))
        print(f'loading {per_ex} out of {tot_ex} training data')
        tot_ex = per_ex

    tot_tf = int(ceil(tot_ex/cfg["tfrec_size"]))  # total tfrecords
    return tot_tf


if __name__ == '__main__':
    if cfg["perc_data"] < 1.0:
        splits = ['train']
    else:
        splits = ['train','val','test']

    for split in ['train']:
        with timebudget(f'TFREC {split}'):
            # get total tfrecords to be created
            tot_tf = get_tot_tf(cfg, split)

            # create save directory if not exist
            base_dir = os.path.join(cfg["out_dir"], cfg["name"], cfg["tfrec_dir"])
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)

            # create vector_fix directory if saving fix gdf and if it not exist
            if not cfg["load_fix"]:
                vector_fix_dir = os.path.join(cfg["out_dir"], f's{cfg["stride"]}', 'vector_fix')
                if not os.path.isdir(vector_fix_dir):
                    os.makedirs(vector_fix_dir)
            
            # basename of tfrec file
            base_fn = os.path.join(base_dir, split)

            args = []
            for proc_idx in range(tot_tf):
                args.append((proc_idx, base_fn, split, cfg))

            # serial implement
            # for _args in args: create_tfrecord_parallel(*_args)
            
            proc_pool = Pool(4)
            proc_pool.starmap(create_tfrecord_parallel, args)

    # save config
    cfg_fn = os.path.join(base_dir, 'cfg.json')
    with open(cfg_fn, 'w') as f:
        json.dump(cfg, f)

    print('config saved!')
