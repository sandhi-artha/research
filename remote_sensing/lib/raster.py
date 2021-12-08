import numpy as np
from PIL import Image
import rasterio as rs
from shapely.geometry import box
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from rasterio import plot
"""TODO : create functions with less dependencies
"""




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

def get_tile_bounds(fns):
    """read raster's bounds with rasterio
    -------
    return: list of tile_bounds
    """
    return [list(rs.open(fn).bounds) for fn in fns]

def show_tile_bounds(bound_list, return_gdf=False):
    """plot or return gdf of boundaries
    hint: can also use gpd.GesSeries(bound) for a single bound
    """
    geometry = [box(*bound) for bound in bound_list]
    # create geodataframe
    d = {'geometry': geometry}
    gdf = gpd.GeoDataFrame(d, crs='epsg:32631')
    if return_gdf:
        return gdf
    else:
        gdf.plot(color='grey', edgecolor='black', alpha=0.8)

def get_bound_inter(split_bound, raster_bound):
    """input: shapely
    returns: geodataframe of intersection
    TODO : auto recognize if it's shapely or plain geometry list
    hint: can also just use split_bound.intersection(raster_bound)
    """
    s_gdf = gpd.GeoDataFrame({'geometry': [split_bound]}, crs='epsg:32631')
    r_gdf = gpd.GeoDataFrame({'geometry': [raster_bound]}, crs='epsg:32631')
    crop_gdf = r_gdf.intersection(s_gdf)
    return 

def create_patch(bounds, lw=1, c='r', ax=None):
    """create a rectangle with given bounds
    bounds = list
        [left, bot, right, top], default bounds used in rasterio
    lw = int
        linewidth, thickness of rectangle
    c = str
        color of line
    ax = matplotlib axis
        if not provided, returns the patch object
    """
    left, bot, right, top = bounds
    width = right-left
    height = top-bot
    rect = patches.Rectangle((left, bot), width, height,
                             linewidth=lw, edgecolor=c, facecolor='none')
    if ax:
        ax.add_patch(rect)
    else:
        return rect

def coor_to_index(bounds, dst_prj):
    """
    start from x0,y0 -> left, bot
    end at x1,y1 -> right, top
    returns : list [left, bot, right, top]
        bounds in image space using the destination projection
    hint: for SN6 SLC, since raster is vflip and rot90 -> r0>r1 (swap them)
        for windowing in rasterio use: ((r0,r1),(c0,c1))
    """
    x0, y0, x1, y1 = bounds
    r0, c0 = rs.transform.rowcol(dst_prj, x0, y0)
    r1, c1 = rs.transform.rowcol(dst_prj, x1, y1)
    return [r0, c0, r1, c1]

def show_16_tiles(fns):
    f = plt.figure(figsize=(16,16))
    for i,fn in enumerate(fns):
        raster = rs.open(fn)
        image = raster.read([1,4,3])
        image = rs.plot.reshape_as_image(image)
        image = np.nan_to_num(image, nan=np.nanmin(image))
        
        ax = f.add_subplot(4,4,i+1)
        ax.imshow(normalize(image))
        ax.set_title(fn)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()