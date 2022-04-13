import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    params:
        @flow: np.array(h, w, 2)
        @confidence: np.array(h, w)
        @K: np.array(3, 3)
        @ep: (3,1) with the last element to be 1, [Vx/Vz, Vx/Vz, 1]
    return value:
        depth_map: np.array(h, w)
    """

    depth_map = np.zeros_like(confidence)
    w = flow.shape[1]
    h = flow.shape[0]
    # x_left_end = -np.ceil(w/2)
    # x_right_end = np.floor(w/2)
    # y_up_end = -np.ceil(h/2)
    # y_down_end = np.floor(h/2)

    # compute pixel coordinate, now the pixel coordinate goes to upper left corner? 
    # xp, yp = np.meshgrid(np.arange(x_left_end,x_right_end), np.arange(y_up_end,y_down_end))
    xp, yp = np.meshgrid(np.arange(0, w), np.arange(0, h))

    # compute calibrated coordinate x = K_-1 @ [xp, yp, 1].T
    homo_pixel_coord = np.vstack((
        xp.flatten(), yp.flatten(), np.ones_like(xp).flatten() 
    ))
    calibrated_coord = np.linalg.inv(K) @ homo_pixel_coord
    p = calibrated_coord[0:2] # (2, n)

    # compute calibrated optical flow
    u = flow[:,:,0].flatten()
    v = flow[:,:,1].flatten()

    calibrated_velocity = np.linalg.inv(K) @ np.vstack((
        u, v, np.zeros_like(u)
    ))
    p_dot_trans = calibrated_velocity[0:2] # (2,n)

    # compute calibrated epipole 
    ep = ep.flatten()/ep[2]
    ep_calibrated = np.linalg.inv(K) @ ep.reshape(-1,1)
    ep_calibrated = ep_calibrated[0:2]

    # compute depth
    Z = np.linalg.norm(p - ep_calibrated, axis = 0)/ np.linalg.norm(p_dot_trans, axis = 0) # (n,)
    depth_map = Z.reshape(h,w)
    depth_map[confidence <= thres] = 0

    """
    STUDENT CODE BEGINS
    """
    truncated_depth_map = np.maximum(depth_map, 0)
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    # You can change the depth bound for better visualization if your depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    print(f'depth bound: {depth_bound}')

    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()
    """
    STUDENT CODE ENDS
    """
    return truncated_depth_map
