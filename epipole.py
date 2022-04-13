import numpy as np
def epipole(u,v,smin,thresh,num_iterations = 1000):
    ''' Takes flow (u,v) with confidence smin and finds the epipole using only the points with confidence above the threshold  
        (for both sampling and finding inliers)
        u, v and smin are (w,h), thresh is a scalar
        output should be best_ep and inliers, which have shapes, respectively (3,) and (n,) 
    '''

    """YOUR CODE HERE -- You can do the thresholding outside the RANSAC loop here
    """
    w = u.shape[1]
    h = u.shape[0]
    u_confident = u[smin>thresh].flatten()
    v_confident = v[smin>thresh].flatten()
    # x, y is referred in pixel coordinate with origin at the center, x-axis points rightwards, y-axis points downwards
    x_left_end = -np.ceil(w/2)
    x_right_end = np.floor(w/2)
    y_up_end = -np.ceil(h/2)
    y_down_end = np.floor(h/2)
    x, y = np.meshgrid(np.arange(x_left_end,x_right_end), np.arange(y_up_end,y_down_end))
    x_confident = x[smin>thresh].flatten()
    y_confident = y[smin>thresh].flatten()

    # these are the indices at which smin>threshold, used to pass gradescope test
    indice_confident = np.arange(0, len(x.flatten()))[smin.flatten()>thresh]
    """ END YOUR CODE
    """
    sample_size = 2

    eps = 10**-2

    best_num_inliers = -1
    best_inliers = None
    best_ep = None

    for i in range(num_iterations):
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(0,np.sum((smin>thresh))))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]

        """YOUR CODE HERE
        """
        xp_sample = x_confident[sample_indices]
        yp_sample = y_confident[sample_indices]
        u_sample = u_confident[sample_indices]
        v_sample = v_confident[sample_indices]

        xp_test = x_confident[test_indices]
        yp_test = y_confident[test_indices]
        u_test = u_confident[test_indices]
        v_test = v_confident[test_indices]

        x_test = np.hstack((
            xp_test.reshape(-1,1), 
            yp_test.reshape(-1,1),
            np.ones((len(xp_test), 1))
        )) # (n, 3)
        optFlow_test = np.hstack((
            u_test.reshape(-1,1), 
            v_test.reshape(-1,1),
            np.zeros((len(xp_test), 1))
        )) # (n, 3)

        # construct the model: calculate the epipole
        x_sample = np.array([
            [xp_sample[0], yp_sample[0], 1],
            [xp_sample[1], yp_sample[1], 1]
        ])
        optFlow_sample = np.array([
            [u_sample[0], v_sample[0], 0],
            [u_sample[1], v_sample[1], 0]
        ])

        # a1 = np.cross(x_sample[0], optFlow_sample[0])
        # a2 = np.cross(x_sample[1], optFlow_sample[1])

        A = np.cross(x_sample,optFlow_sample)
        U, S, VT = np.linalg.svd(A)
        ep = VT[-1].reshape(-1,1) # (3,1)

        # count inliers: e.T @ (x cross optFlow) < epsi
        x_cross_optFlow = np.cross(x_test, optFlow_test) # row vector is the crossed vector (n,3)
        distance = np.abs(ep.T @ x_cross_optFlow.T) # (1,3) @ (3,n)
        distance = distance.flatten()
        test_inliers = test_indices[distance<eps]
        sample_test_inliers = np.hstack((sample_indices, test_inliers))
        inliers = indice_confident[sample_test_inliers] 
        """ END YOUR CODE
        """

        #NOTE: inliers need to be indices in original input (unthresholded), 
        #sample indices before test indices for the autograder
        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_ep = ep
            best_inliers = inliers

    return best_ep.flatten(), best_inliers