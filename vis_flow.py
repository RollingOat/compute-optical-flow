import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    params:
        @img: np.array(h, w)
        @flow_image: np.array(h, w, 2)
        @confidence: np.array(h, w)
    return value:
        None
    """
    h = image.shape[0]
    w = image.shape[1]
    x, y = np.meshgrid(np.arange(0,w), np.arange(0, h))
    x_good = x[confidence>threshmin]
    y_good = y[confidence>threshmin]
    flow_x = np.where(confidence>threshmin, flow_image[:,:,0], 0)
    flow_y = np.where(confidence>threshmin, flow_image[:,:,1], 0)
    
    plt.quiver(x, y, (flow_x*10).astype(int), (flow_y*10).astype(int), 
                angles='xy', scale_units='xy', scale=1., color='red', width=0.001)
    plt.imshow(image)
    return



