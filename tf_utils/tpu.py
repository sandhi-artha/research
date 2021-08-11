import numpy as np
import re
import tensorflow as tf
from functools import partial # used when parsing tfrecords


# tfrecord features
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor. intended for the image data
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



# tpu training
def count_data_items(filenames):
    """the number of data items is written in the name of the .tfrec files, 
    i.e. test10-687.tfrec = 687 data items"""
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def load_dataset(filenames, read_tfrecord, ordered=False):
    """
    takes list of .tfrec files, read using TFRecordDataset,
    parse and decode using read_tfrecord func which return features
    use ordered=True for test dataset where submission order is important
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.with_options(ignore_order) 
    dataset = dataset.map(partial(read_tfrecord), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset