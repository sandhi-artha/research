import numpy as np
from PIL import Image
# import rasterio as rs
# from rasterio import plot





def read_slc(path):
    Image.MAX_IMAGE_PIXELS = None  # surprass the DecompressionBombWarning
    slc = Image.open(path, 'r')
    slc = np.array(slc, dtype=np.uint8)
    # slc = rs.open(path)
    # slc = slc.read()
    # slc = plot.reshape_as_image(slc)
    return slc

def get_inset(img, aoi, c=None):
    (x0, x1), (y0, y1) = aoi
    if len(img.shape) == 2:
        return img[x0:x1, y0:y1]
    else:
        if c == None:  # if channel not specified, return all
            c = range(img.shape[-1])
        return img[x0:x1, y0:y1, c]

def fix_orient(image):
    image = np.flipud(image)
    return np.rot90(image)