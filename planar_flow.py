import numpy as np

def compute_planar_params(flow_x, flow_y, K,
                                up=[256, 0], down=[512, 256]):
    """
    params:
        @flow_x: np.array(h, w)
        @flow_y: np.array(h, w)
        @K: np.array(3, 3)
        @up: [y_start, x_start]
        @down: [y_end, x_end]
    return value:
        sol: np.array(8,)
    """
    h = flow_x.shape[0]
    w = flow_x.shape[1]

    # the left,right,upper,lower limit of the left bottom
    x_left = up[1]
    x_right = down[1]
    y_up = up[0]
    y_down = down[0] 
    
    
    # compute pixel coordinate
    left_bottom_xp,left_bottom_yp = np.meshgrid(np.arange(x_left,x_right), np.arange(y_up,y_down))
    num_of_pixel = len(left_bottom_xp.flatten())
    pixel_coord = np.vstack((
        left_bottom_xp.flatten(),
        left_bottom_yp.flatten(),
        np.ones(num_of_pixel)
    ))

    # compute calibrated coordinate
    calibrated_coord = np.linalg.inv(K) @ pixel_coord # (3,n)

    # compute calibrated flow (in calibrated coordinate) of the left bottom part of image
    flow_x = flow_x[left_bottom_yp, left_bottom_xp]
    flow_y = flow_y[left_bottom_yp, left_bottom_xp]
    flow_pixel = np.vstack((
        flow_x.flatten(),
        flow_y.flatten(),
        np.zeros(num_of_pixel)
    ))
    calibrated_flow = np.linalg.inv(K) @ flow_pixel # (3,n)

    # construct Ax = b to solve for a1,...,a8
    for i in range(num_of_pixel):
        x = calibrated_coord[0,i]
        y = calibrated_coord[1,i]
        a = np.array([
            [x**2, x*y, x, y, 1, 0, 0, 0],
            [x*y, y**2, 0 ,0, 0, y, x, 1]
        ])
        x_dot = calibrated_flow[0,i]
        y_dot = calibrated_flow[1,i]
        b = np.array([x_dot,y_dot]).reshape(-1,1)
        if i == 0:
            A = a
            B = b
        else:
            A = np.vstack((
                A, a
            ))
            B = np.vstack((
                B, b
            ))
    
    # solve Ax=b
    sol = np.linalg.lstsq(A,B,rcond = None)[0]

    return sol.flatten()
    
