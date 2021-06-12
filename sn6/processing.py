import numpy as np

def norm(plane):
    """make sure if max val exceeds given 92.88, it won't result in value >255
    and making it become 0 when converted to uint8 bcz of overflow
    
    used only for sar images since their values are floats
    
    Parameters:
    -----------
    plane: numpy array of any size and dimension
    """
    max_val = plane.max() if plane.max()>92.88 else 92.88
    plane = plane / max_val * 255
    return plane.astype(np.uint8)

def get_region_index(raster):
    """
    leaves nodata trails in the edges
    np.argwhere() returns list of index. [[row,col],[row,col]]
        of every element that gets true condition
    """
    bin_mask = raster.read_masks(1)
    coords = np.argwhere(bin_mask==255)
    row0,col0 = coords.min(axis=0)  # find lowest row and col
    row1,col1 = coords.max(axis=0)  # find highest row and col
    return row0,row1,col0,col1

def mask2contour(mask, width=1):
    """Converts a polygon to only its edges, for ploting

    Parameters:
    -----------
    mask : (w,h,1)
        use np.squeeze(mask, axis=2).astype('uint8') on mask

    width : int
        how thick the lines
    """
    # convert to uint8 if not
    mask = np.squeeze(mask, axis=2).astype('uint8')
    mask = mask.astype('uint8')

    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3)