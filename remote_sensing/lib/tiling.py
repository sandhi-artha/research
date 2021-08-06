import numpy as np


# snippet from sn6_create, grab index that shows true pixels
def get_region_index(bin_mask):
    coords = np.argwhere(bin_mask==1)
    row0,col0 = coords.min(axis=0)  # find lowest row and col
    row1,col1 = coords.max(axis=0)  # find highest row and col
    return ((row0,row1),(col0,col1))

# tiling algorithm
def get_tile_index(res, stride, bound):
    tile_index_list = []  # not the best way to store images, but just as prototype
    (h_str, h_end), (w_str, w_end) = bound

    min_len = res*0.5  # length must be larger than this to be used
    i = 0  # row index
    while i<100:  # row/height
        r0 = i*(res-stride) + h_str
        r1 = r0+res
        j = 0  # col index

        # conditional row break
        if r1 >= h_end:  # if last row of tile exceeds or equal to h
            if h_end-r0 < min_len:
                # if the remaning portion is smaller than half of res, exit
                break

        while j<100:  # column/width
            c0 = j*(res-stride) + w_str
            c1 = c0+res

            # conditional col break
            if c1 >= w_end:  # if the end exceeds or equal to w
                if w_end-c0 < min_len:
                    # if the remaning portion is smaller than half of res, exit
                    break

            tile_index_list.append(((r0,r1),(c0,c1)))
            j += 1
        
        i += 1

    return tile_index_list

def tile_image_mask(image, c, mask, tile_idx, bound):
    """
    image: w,h,c np array
    c: array indicating which pol to use
    mask: w,h binary np array
    tile_idx and bound: ((r0,r1),(c0,c1))
    """
    (r0,r1), (c0,c1) = tile_idx
    (h_str, h_end), (w_str, w_end) = bound

    res = r1-r0

    # for tile index going outside boundaries. to right (c1), and to bot (r1)
    if c1 >= w_end or r1>=h_end:
        # create np as place_holder
        tile = np.zeros((res,res,len(c)),dtype=np.uint8)
        tile_mask = np.zeros((res,res),dtype=np.uint8)

        # determine the end of col and row
        # c1 and r1 can go over the image and cause error, so we take w_end and h_end in that case
        col_end = np.min([c1, w_end])
        row_end = np.min([r1, h_end])

        # get the partial raster
        par = image[r0:row_end, c0:col_end, c]  # if end of col, grab from c0 until the width's end 
        par_mask = mask[r0:row_end, c0:col_end]
        
        # fill the big empty array with the small partial arr
        tile[0:par.shape[0], 0:par.shape[1], 0:par.shape[2]] = par
        tile_mask[0:par_mask.shape[0], 0:par_mask.shape[1]] = par_mask

    else:
        tile = image[r0:r1, c0:c1, c]  # crop image
        tile_mask = mask[r0:r1, c0:c1]  # crop mask

    return tile, tile_mask