import numpy as np

from .viz import show_stats

def to_whc(image):
    """swaps channel/band to last [width, height, channel]
    --------
    image: np.array of [c,w,h] format
    """
    return np.transpose(image,[1,2,0])


def to_cwh(image):
    """swaps channel/band to first [channel, width, height]
    --------
    image: np.array of [w,h,c] format
    """
    return np.transpose(image,[2,0,1])

def _norm_plane(plane):
    """normalization only on 1 plane
    plane: np.array in format [w,h,c]
    """
    plane =  plane - np.min(plane)
    return plane / np.max(plane)

def normalize(plane, cfirst=False):
    """Scales pixel value to 0.0 - 1.0
    does not change image shape
    --------
    plane: np.array in format [w,h,c]
    cfirst: bool, use cfirst=1 if image format [c,w,h]
    """
    if cfirst: plane = to_whc(plane)

    # 1ch image usually have only [w,h] or [w,h,1]
    if len(plane.shape) == 2 or plane.shape[-1] == 1:
        norm_image = _norm_plane(plane)
    else:
        norm_image = np.zeros(plane.shape)
        for i in range(plane.shape[-1]):
            norm_image[:,:,i] = _norm_plane(plane[:,:,i])

    # swap back to original ch first
    if cfirst: norm_image = to_cwh(norm_image)

    return norm_image

def _stan_plane(plane):
    plane = plane - np.mean(plane)
    return plane / np.std(plane)

def standardize(plane, cfirst=False):
    """Scales pixel value to have mean=0.0 and std=1.0
    does not change image shape
    --------
    plane: np.array in format [w,h,c]
    cfirst: bool, use cfirst=1 if image format [c,w,h]
    """
    if cfirst: plane = to_whc(plane)

    # 1ch image usually have only [w,h] or [w,h,1]
    if len(plane.shape) == 2 or plane.shape[-1] == 1:
        stan_image = _stan_plane(plane)
    else:
        stan_image = np.zeros(plane.shape)
        for i in range(plane.shape[-1]): 
            stan_image[:,:,i] = _stan_plane(plane[:,:,i])

    # swap back to original ch first
    if cfirst: stan_image = to_cwh(stan_image)

    return stan_image

def test_lib():
    """test function
    """
    a1 = np.random.randint(0,10,[40,40,3])
    a2 = to_cwh(a1)

    # ch last
    show_stats(a1)
    a11 = normalize(a1)
    show_stats(a11)

    # ch first
    show_stats(a2,1)
    a21 = normalize(a2,1)
    show_stats(a21,1)