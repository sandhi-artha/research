import os
import json
import glob
import pickle
from multiprocessing import Pool

from numpy import ceil
from timebudget import timebudget

from dataset_cfg import post_cfg
from dataset_gen import load_timestamps, run_parallel_op
from lib.tfrec import create_tfrecord_parallel, get_tile_paths

"""
This script expects tiles already created and saved in {out_dir}/{name},
will filter tiles based selected SLC stripe
"""

def get_tot_tf(cfg, split):
    if split=='train':
        timestamps = load_timestamps(post_cfg['perc_data'])
    else:  # for val and train, take all SLC timestamps
        timestamps = load_timestamps(1)

    tot_ex = 0
    for ts in timestamps:
        path = os.path.join(cfg["out_dir"], cfg["name"], 'raster', f'*{ts}*{split}*.tif')
        tot_ex += len(glob.glob(path))    # total examples

    tot_tf = int(ceil(tot_ex/post_cfg["tfrec_size"]))  # total tfrecords
    print(f'creating {tot_tf} tfrecs out of {tot_ex} {split} tiles')
    return tot_tf


if __name__ == '__main__':
    # load cfg in raster folder
    with open(os.path.join(post_cfg['tile_dir'], 'cfg.json')) as f:
        cfg = json.load(f)

    for split in post_cfg['splits']:
        with timebudget(f'TFREC {split}'):
            # get total tfrecords to be created
            tot_tf = get_tot_tf(cfg, split)

            # create save directory if not exist
            base_dir = os.path.join(cfg["out_dir"], cfg["name"], post_cfg["tfrec_dir"])
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)

            # create vector_fix directory if saving fix gdf and if it not exist
            if not post_cfg["load_fix"]:
                vector_fix_dir = os.path.join(cfg["out_dir"], f's{cfg["stride"]}', 'vector_fix')
                if not os.path.isdir(vector_fix_dir):
                    os.makedirs(vector_fix_dir)
            
            # basename of tfrec file
            base_fn = os.path.join(base_dir, split)

            args = []
            for proc_idx in range(tot_tf):
                args.append((proc_idx, base_fn, split, cfg, post_cfg))

            run_parallel_op(create_tfrecord_parallel, args, post_cfg['tfrec_proc'], debug=0)


    # save configs
    cfg_fn = os.path.join(base_dir, 'cfg.json')
    with open(cfg_fn, 'w') as f:
        json.dump(cfg, f)
    with open(cfg_fn.replace('cfg','post_cfg'), 'w') as f:
        json.dump(post_cfg, f)
    print('configs saved!')
