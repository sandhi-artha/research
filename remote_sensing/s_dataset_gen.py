import os
import random
import pickle
from multiprocessing import Pool

from numpy import ceil
from timebudget import timebudget

from dataset_cfg import cfg
from sar_preproc import SarPreproc
from tile_scheme import parallel_tile_generator

def create_s_fn():
    full_raster_path = '../../dataset/sensor/base/raster'
    s_train_fn = os.listdir(full_raster_path)
    # filter only 'train'
    s_train_fn = [fn for fn in s_train_fn if 'train' in fn]
    # sort and shuffle
    s_train_fn.sort()
    random.Random(17).shuffle(s_train_fn)
    # take percentage of it
    tot_len = len(s_train_fn)
    per_len = int(ceil(tot_len*cfg["perc_data"]))
    s_train_fn = s_train_fn[:per_len]

    # save shuffled list
    with open('s_fn.pickle', 'wb') as f:
        pickle.dump(s_train_fn, f)

def create_s_list():
    with open('s_fn.pickle', 'rb') as f:
        s_train_fn = pickle.load(f)  # len 1757

    # get unique timestamps
    timestamps = []
    for fn in s_train_fn:
        sp = fn.split('_')
        timestamps.append(f'{sp[1]}_{sp[2]}')

    timestamps = set(timestamps)  # len 202 (all slices)
    s_list = []
    # sensor_20190804111224_20190804111453_o1_base_test_s0_0000.tif
    # [['20190804111224_20190804111453', ['0000','0020', ...]], ...]
    for ts in timestamps:
        tile_ids = [fn.split('.')[0][-4:] for fn in s_train_fn if ts in fn]
        tile_ids.sort()
        s_list.append([ts, tile_ids])

    with open('s_list.pickle','wb') as f:
        pickle.dump(s_list, f)

def create_s_scheme():
    """
    reads the timestamps and tile_ids from 25% train data,
        filters tile_scheme to include these 25% images
        saves the filtered scheme as 's_tile_scheme'
    """
    with open('s_list.pickle', 'rb') as f:
        s_list = pickle.load(f)

    full_sch_dir = '../../dataset/sensor/s0/tile_scheme'
    save_sch_dir = '../../dataset/sensor/s0/s_tile_scheme'
    train_sch_path = os.listdir(full_sch_dir)
    train_sch_path = [sch for sch in train_sch_path if 'train' in sch]

    # filter train_sch_path contained in timestamp
    for timestamp, tile_ids in s_list:
        for sch_path in train_sch_path:
            if timestamp in sch_path:  # if timestamp is contained in the path
                load_fn = os.path.join(full_sch_dir, sch_path)
                # load tile_schemes containing: [[name, bound, profile],..]
                with open(load_fn, 'rb') as f:
                    tile_schemes = pickle.load(f)
                
                # filter elements of tile_scheme contained in tile_ids
                s_tile_scheme = []
                for tile_id in tile_ids:
                    for tile_scheme in tile_schemes:  
                        if tile_id in tile_scheme[0]:  # if tile_scheme contained in name (1st element)
                            s_tile_scheme.append(tile_scheme)

                # save the filtered tile_schemes
                save_fn = os.path.join(save_sch_dir, timestamp)
                with open(save_fn, 'wb') as f:
                    pickle.dump(s_tile_scheme, f)




if __name__=='__main__':
    # create_s_fn()
    # create_s_list()
    # create_s_scheme()

    save_path = os.path.join(cfg["out_dir"], cfg["name"]) # where raster folder is saved
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    s_scheme_dir = '../../dataset/sensor/s0/s_tile_scheme'
    timestamps = os.listdir(s_scheme_dir)
    with timebudget('25% train tiling'):
        for i, timestamp in enumerate(timestamps):

            out_fn = 'output.tif'
            sar_preproc = SarPreproc(cfg, timestamp, cfg["in_dir"], cfg["out_dir"], out_fn)
            with timebudget(f'{timestamp}, {i} of {len(timestamps)}'):
                sar_preproc()

            sch_path = os.path.join(s_scheme_dir, timestamp)
            with open(sch_path, 'rb') as f:
                tile_schemes = pickle.load(f)

            proc_slc_path = os.path.join(cfg["out_dir"], out_fn)  # output.tif path

            schemes = []
            for tile_scheme in tile_schemes:
                schemes.append((tile_scheme, proc_slc_path, save_path))

            n_tile = len(tile_schemes)
            n_proc = int(ceil(n_tile/2))
            print(f'tiling with {n_proc} process')
            proc_pool = Pool(n_proc)
            proc_pool.starmap(parallel_tile_generator, schemes)
