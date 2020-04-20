import numpy as np
from scipy import signal
import cv2

def lucas_kanade_opt_flow(I1, I2, window_size, tau=1e-2):
 
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    w = window_size//2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1 = I1 / 255. # normalize pixels
    I2 = I2 / 255. # normalize pixels
    mode = 'same'
    fx = signal.convolve2d(I1, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1, kernel_y, boundary='symm', mode=mode)
    ft = I2 - I1
    u = np.zeros(I1.shape)
    v = np.zeros(I1.shape)
    # within window window_size * window_size
    for i in range(w, I1.shape[0]-w):
        for j in range(w, I1.shape[1]-w):
            Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten().reshape(-1,1)
            Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten().reshape(-1,1)
            It = ft[i-w:i+w+1, j-w:j+w+1].flatten().reshape(-1,1)
            S = np.concatenate((Ix, Iy), axis = 1)

            U,D,V = np.linalg.svd(S.T.dot(S))
            if np.min(D) < tau or 0 in D:
            	u[i,j] = 0
            	v[i,j] = 0
            else:	
	            nu = np.linalg.inv(S.T.dot(S)).dot(S.T).dot(-1*It)
	            u[i,j]=nu[0]
	            v[i,j]=nu[1]
    flow = np.zeros((I1.shape[0],I1.shape[1],2))
    flow[:,:,0] = u
    flow[:,:,1] = v
    return flow
