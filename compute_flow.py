import numpy as np
import pdb

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    compute the optical flow of a (5,5) patch with (x,y) as the center.
    Solve Ix * u + Iy * v + It = 0
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
        @x: int
        @y: int
    return value:
        flow: np.array(2,)
        conf: np.array(1,)
    """
    h = Ix.shape[0]
    w = Ix.shape[1]
    # find its neighboring pixel indices
    # neighbor_y = np.arange(int(y-(size-1)/2),int(y+(size-1)/2)+1)
    # neighbor_x = np.arange(int(x-(size-1)/2),int(x+(size-1)/2)+1)

    neighbor_y = np.array([y-2, y-1, y, y+1, y+2])
    neighbor_x = np.array([x-2, x-1, x, x+1, x+2])
    col_indices, row_indices = np.meshgrid(neighbor_x, neighbor_y)

    # only take the indices within boudary
    valid_indices_mask = np.logical_and (np.logical_and(row_indices>=0 , row_indices<h), np.logical_and(col_indices>=0 , col_indices<w))
    valid_row_indices = row_indices[valid_indices_mask]
    valid_col_indices = col_indices[valid_indices_mask]
    Ix_patch = Ix[valid_row_indices, valid_col_indices].reshape(-1,1) # (25 or 9 or 15, 1)
    Iy_patch = Iy[valid_row_indices, valid_col_indices].reshape(-1,1)
    It_patch = It[valid_row_indices, valid_col_indices].reshape(-1,1)

    # construct a linear system Ax=b
    A = np.hstack((Ix_patch, Iy_patch))
    b = -It_patch

    # solve the lienar system
    solution, _, _, s = np.linalg.lstsq(A, b, rcond=None)
    # solution = np.linalg.inv(A.T @ A) @ A.T @ b
    conf = np.min(s)
    flow = solution.flatten()
    
    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    params:
        @Ix: np.array(h, w)
        @Iy: np.array(h, w)
        @It: np.array(h, w)
    return value:
        flow: np.array(h, w, 2)
        conf: np.array(h, w)
    """
    image_flow = np.zeros([Ix.shape[0], Ix.shape[1], 2])
    confidence = np.zeros([Ix.shape[0], Ix.shape[1]])
    for x in range(Ix.shape[1]):
        for y in range(Ix.shape[0]):
            flow, conf = flow_lk_patch(Ix, Iy, It, x, y)
            image_flow[y, x, :] = flow
            confidence[y, x] = conf
    return image_flow, confidence

    

