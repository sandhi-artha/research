import pickle
import os
import argparse

import numpy as np
from sklearn.cluster import KMeans

from dataset_cfg import cfg
from lib.raster import get_tile_bounds

def get_timestamps(to_list, orient=None):
    # use 2 or integers other than 0,1 to get both orients
    if orient==0:
        to_list_filt = [to[:-2] for to in to_list if to[-1] == '0']
    elif orient==1:
        to_list_filt = [to[:-2] for to in to_list if to[-1] == '1']
    else:
        to_list_filt = [to[:-2] for to in to_list]
    
    to_list_filt.sort()
    return to_list_filt

def get_slc_path(timestamp):
    fn = f'CAPELLA_ARL_SM_SLC_HH_{timestamp}.tif'
    return os.path.join(cfg['in_dir'], fn)

def get_sample_in_clusters(n_clusters, arr):
    # create n clusters and 
    kmeans = KMeans(n_clusters=n_clusters)
    y_kmeans = kmeans.fit_predict(arr)

    # grab 1 sample_idx from each cluster
    s_idx = []
    for cluster in range(n_clusters):
        # get indexes belonging to a cluster
        s = np.argwhere(y_kmeans == cluster)
        # choose 1, returns list, so get [0]
        s_idx.append(np.random.choice(s.flatten(),1)[0])
    
    return s_idx


if __name__=='__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--orient', type=int, required=True)
    parser.add_argument('-n', '--n_clusters', type=int, required=True)
    args = parser.parse_args()
    orient = args.orient
    n = args.n_clusters

    with open('timestamp_orientation.pickle','rb') as f:
        to_list = pickle.load(f)
    
    timestamps = get_timestamps(to_list, orient=orient)
    slc_paths = [get_slc_path(ts) for ts in timestamps]
    slc_bounds = get_tile_bounds(slc_paths)
    
    print(f'grabbing {n} samples from {len(slc_paths)} of orientation {orient}')

    # mid = top-bot/2 + bot
    mid = [((b[3]-b[1])/2) + b[1] for b in slc_bounds]

    # convert to np.array and add 2nd dim
    mid_arr = np.expand_dims(np.array(mid), axis=-1)
    
    # get sample indexes
    s_idx = get_sample_in_clusters(n, mid_arr)

    # grab sample slc_paths
    sample_slc_paths = [slc_paths[i] for i in s_idx]
    out_fn = f'sample_{n}_{orient}.pickle'
    print(f'saving to {out_fn}')
    with open(out_fn, 'wb') as f:
        pickle.dump(sample_slc_paths, f)
