import tensorflow as tf
import glob
from zipfile import ZipFile


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


def zip_kaggle(path, name):
    # NEEDS TESTING!
    """zips all files in the path ready for kaggle upload"""
    file_paths = glob.glob(path)
    with ZipFile(name, 'w') as z:
        for fp in file_paths:
            z.write(fp)
    
    print('zip success!')
