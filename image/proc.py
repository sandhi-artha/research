import numpy as np

from .viz import show_stats

def to_hwc(image):
    """swaps channel/band to last [height, width, channel]
    --------
    image: np.array of [c,h,w] format
    """
    return np.transpose(image,[1,2,0])


def to_chw(image):
    """swaps channel/band to first [channel, height, width]
    --------
    image: np.array of [h,w,c] format
    """
    return np.transpose(image,[2,0,1])

def _norm_plane(plane):
    """normalization only on 1 plane
    plane: np.array in format [h,w,c]
    """
    plane =  plane - np.nanmin(plane)
    return plane / np.nanmax(plane)

def normalize(plane, cfirst=False):
    """Scales pixel value to 0.0 - 1.0
    does not change image shape
    --------
    plane: np.array in format [h,w,c]
    cfirst: bool, use cfirst=1 if image format [c,h,w]
    """
    if cfirst: plane = to_hwc(plane)

    # 1ch image usually have only [w,h] or [w,h,1]
    if len(plane.shape) == 2 or plane.shape[-1] == 1:
        norm_image = _norm_plane(plane)
    else:
        norm_image = np.zeros_like(plane, dtype=np.float32)
        for i in range(plane.shape[-1]):
            norm_image[:,:,i] = _norm_plane(plane[:,:,i])

    # swap back to original ch first
    if cfirst: norm_image = to_chw(norm_image)

    return norm_image

def _stan_plane(plane):
    plane = plane - np.mean(plane)
    return plane / np.std(plane)

def standardize(plane, cfirst=False):
    """Scales pixel value to have mean=0.0 and std=1.0
    does not change image shape
    --------
    plane: np.array in format [h,w,c]
    cfirst: bool, use cfirst=1 if image format [c,h,w]
    """
    if cfirst: plane = to_hwc(plane)

    # 1ch image usually have only [w,h] or [w,h,1]
    if len(plane.shape) == 2 or plane.shape[-1] == 1:
        stan_image = _stan_plane(plane)
    else:
        stan_image = np.zeros(plane.shape)
        for i in range(plane.shape[-1]): 
            stan_image[:,:,i] = _stan_plane(plane[:,:,i])

    # swap back to original ch first
    if cfirst: stan_image = to_chw(stan_image)

    return stan_image

def test_lib():
    """test function
    """
    a1 = np.random.randint(0,10,[40,40,3])
    a2 = to_chw(a1)

    # ch last
    show_stats(a1)
    a11 = normalize(a1)
    show_stats(a11)

    # ch first
    show_stats(a2,1)
    a21 = normalize(a2,1)
    show_stats(a21,1)