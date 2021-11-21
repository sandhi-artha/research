from dataset_cfg import cfg

from .lib import tfrec

in_path = '/content/drive/MyDrive/spacenet6/solaris_proc/base'
for split in ['train','val','test']:
    fp = tfrec.get_tile_paths(in_path, split)
    base_fn = f'tfrecord/{split}'
    tfrec.create_tfrecord(*fp, cfg, base_fn)