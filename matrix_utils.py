"""
Simple Matrix Utilities
stough, 202-
"""

import numpy as np
from skimage.util import img_as_ubyte
from scipy.interpolate import interp1d


# Given an ndarray, returns a tuple of info...
def arr_info(arr):
    """
    arr_info(arr): return a tuple of shape, type, min, and max for the ndarray arr.
    """
    return arr.shape, arr.dtype, arr.min(), arr.max()


def make_linmap(inputrange=[0,1], outputrange=[0,255]):
    '''
    make_linmap(inputrange=[0,1], outputrange=[0,255]): return an anonymous function
    linear mapping from the inputrange to the outputrange.
    '''
    a,b = inputrange
    c,d = outputrange
    
    return lambda x: (1-((x-a)/(b-a)))*c + ((x-a)/(b-a))*d

def normalize_ubyte(I):
    '''
    normalize_ubyte(I): return an 8-bit normalized version of the input image I.
    '''
    return np.uint8(255*( (I-I.min()) / (I.max()-I.min() )))


def remove_alpha(img):
    # if the image is RGBA, remove the alpha channel
    if img.shape[2] == 4:
        return img[:,:,:3]
    elif img.shape[2] == 3:
        return img
    else:
        raise TypeError("Image must be RGB or RGBA.")
        
    
    
def preprocess_img(img):
    return normalize_ubyte(remove_alpha(img))


def window_image(img, lower_percentile=20, upper_percentile=80):
    '''
    window_image(img, lower_bound, upper_bound): return an image windowed -- 20%-80% percentile scaled to 0-255.
    '''
    img = remove_alpha(img.copy())
    if img.dtype != np.uint8:
        img = normalize_ubyte(img)
        
    lower_bound,upper_bound = np.percentile(img, (lower_percentile,upper_percentile))
        
    xp = [lower_bound,upper_bound]
    fp = [0,255]
    img = np.uint8(np.interp(img, xp, fp))
    
    return img