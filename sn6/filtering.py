import cv2
import numpy as np
from .processing import norm


### filtering
def get_hamming_filter(image,r):
    """
    image: (w,h)
        must be grayscale, square, dtype as float
    r: int
        larger r produces stronger filtering
    output:
        normalized image in uint8
    """
    # create hamming window with radius r, larger r means more blur
    ham = np.hamming(image.shape[0])[:,None]  # 1D hamming
    ham2d = np.sqrt(np.dot(ham, ham.T))**r  # expand to 2D hamming
    
    # apply fourier transform (applies to float32 type only)
    # output is numpy array of 2 channels: imaginary and real
    f = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
    # shift the image, why?
    f_shifted = np.fft.fftshift(f)
    f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
    # hamming window filter is applied by multipying to the complex representation
    f_filtered = ham2d * f_complex
    
    # shift back the shifted image
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    # get image in spatial domain
    inv_image = np.fft.ifft2(f_filtered_shifted)
    
    # NORMALIZE BACK TO UINT8
    filtered_image = np.abs(inv_image)  # remove negative
    # shift lowest val to zero and max val to 255
    filtered_image = (filtered_image-np.min(filtered_image)) / np.max(filtered_image) * 255

    return filtered_image.astype(np.uint8)

def db2mag(ydb):
    """ follows the formula from matlab
    ymag = 10.^(ydb/20)
    """
    return np.power(10, ydb/20)

def mag2db(ymag):
    # 20*log(ymag)
    return 20*np.log10(ymag)

def get_average_filter(image, r):
    """
    apply average filter in linear scale (magnitude), then convert back to dB
    image: (w,h)
        must be grayscale, square, dtype as float
    r: int
        larger r produces stronger filtering
    output:
        blurred image, normalized in each plane and converted to uint8
    """
    image = db2mag(image)
    image = cv2.blur(image, (r,r))
    image = mag2db(image)
    return norm(image)

def filter_image(filter, image, r):
    if filter=='average':
        return get_average_filter(image, r)
    if filter=='hamming':
        return get_hamming_filter(image, r)