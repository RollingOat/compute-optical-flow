import numpy as np
from scipy.ndimage import convolve1d

"""
STUDENT CODE BEGINS
Define 1D filters as global variables.
"""
# Gaussian kernel
g = np.array([
    0.015625, 0.093750, 0.234375, 0.312500, 0.234375, 0.093750, 0.015625
])
# derivative kernel
K = np.array([-1, 0, 1])
# first derivative of Gaussian kernel
h = np.array(
    [0.03125, 0.12500, 0.15625, 0, -0.15625, -0.1250, -0.03125]
) 


def compute_Ix(imgs):
    """
    params:
        @imgs: np.array(h, w, N)
    return value:
        Ix: np.array(h, w, N)
    """
    smooth_y = convolve1d(imgs, g, axis = 0)
    smooth_yt = convolve1d(smooth_y, g, axis = 2)
    Ix = convolve1d(smooth_yt, h, axis = 1)
    return Ix

def compute_Iy(imgs):
    """
    params:
        @imgs: np.array(h, w, N)
    return value:
        Iy: np.array(h, w, N)
    """
    smooth_x = convolve1d(imgs, g, axis = 1)
    smooth_xt = convolve1d(smooth_x, g, axis = 2)
    Iy = convolve1d(smooth_xt, h, axis = 0)
    return Iy

def compute_It(imgs):
    """
    params:
        @imgs: np.array(h, w, N)
    return value:
        It: np.array(h, w, N)
    """
    smooth_y = convolve1d(imgs, g, axis = 0)
    smooth_xy = convolve1d(smooth_y, g, axis = 1)
    It = convolve1d(smooth_xy, h, axis = 2)
    return It
