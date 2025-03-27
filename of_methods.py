import numpy as np
import cv2
import matplotlib.pyplot as plt
import ex1_utils as utils
from scipy.signal import convolve2d

def lucaskanade(im1, im2, N):
    # Compute spatial derivatives I_x and I_y using Gaussian derivatives
    I_x, I_y = utils.gaussderiv(im1, sigma=1.0)
    
    # Compute temporal derivative I_t
    I_t = im2 - im1
    
    # Define a uniform kernel for the neighborhood
    kernel = np.ones((N, N))
    
    # Compute the sums over the neighborhood using convolution
    I_x2 = convolve2d(I_x**2, kernel, mode='same')
    I_y2 = convolve2d(I_y**2, kernel, mode='same')
    I_xy = convolve2d(I_x * I_y, kernel, mode='same')
    I_xt = convolve2d(I_x * I_t, kernel, mode='same')
    I_yt = convolve2d(I_y * I_t, kernel, mode='same')
    
    # Compute the determinant D
    D = I_x2 * I_y2 - I_xy**2
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    D[D == 0] = epsilon
    
    # Compute the displacement vectors u and v
    u = -(I_y2 * I_xt - I_xy * I_yt) / D
    v = -(I_x2 * I_yt - I_xy * I_xt) / D
    
    return u, v

def hornschunck(im1, im2, n_iters, lmbd):
    # Compute spatial derivatives I_x and I_y using Gaussian derivatives
    I_x, I_y = utils.gaussderiv(im1, sigma=1.0)
    
    # Compute temporal derivative I_t
    I_t = im2 - im1
    
    # Initialize displacement vectors u and v to zero
    u = np.zeros_like(im1)
    v = np.zeros_like(im1)
    
    # Define the Laplacian kernel for averaging
    L_d = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    
    # Iteratively update u and v
    for _ in range(n_iters):
        # Compute the average of the neighboring values
        u_avg = convolve2d(u, L_d, mode='same', boundary='symm')
        v_avg = convolve2d(v, L_d, mode='same', boundary='symm')
        
        # Compute P and D
        P = I_x * u_avg + I_y * v_avg + I_t
        D = lmbd + I_x**2 + I_y**2
        
        # Update u and v
        u = u_avg - I_x * (P / D)
        v = v_avg - I_y * (P / D)
    
    return u, v

def lucaskanade_improved(im1, im2, N):
    # Compute spatial derivatives I_x and I_y using Gaussian derivatives
    I_x, I_y = utils.gaussderiv(im1, sigma=1.0)
    
    # Compute temporal derivative I_t
    I_t = im2 - im1
    
    # Define a uniform kernel for the neighborhood
    kernel = np.ones((N, N))
    
    # Compute the sums over the neighborhood using convolution
    I_x2 = convolve2d(I_x**2, kernel, mode='same')
    I_y2 = convolve2d(I_y**2, kernel, mode='same')
    I_xy = convolve2d(I_x * I_y, kernel, mode='same')
    I_xt = convolve2d(I_x * I_t, kernel, mode='same')
    I_yt = convolve2d(I_y * I_t, kernel, mode='same')
    
    # Compute the determinant D
    D = I_x2 * I_y2 - I_xy**2
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    D[D == 0] = epsilon
    
    # Compute the displacement vectors u and v
    u = -(I_y2 * I_xt - I_xy * I_yt) / D
    v = -(I_x2 * I_yt - I_xy * I_xt) / D
    
    # Define the Harris corner response parameter
    k = 0.04
    
    # Compute the Harris response
    harris_response = (I_x2 * I_y2 - I_xy**2) - k * (I_x2 + I_y2)**2
    
    threshold=0.01
    
    # Threshold the Harris response to filter out unreliable regions
    mask = harris_response > threshold
    
    # Apply the mask to the optical flow vectors
    u[~mask] = 0
    v[~mask] = 0
    
    return u, v

def hornschunck_speedup(im1, im2, n_iters, lmbd, u_init, v_init):
    # Compute spatial derivatives I_x and I_y using Gaussian derivatives
    I_x, I_y = utils.gaussderiv(im1, sigma=1.0)
    
    # Compute temporal derivative I_t
    I_t = im2 - im1
    
    # Initialize displacement vectors u and v to the initial values
    u = u_init
    v = v_init
    
    # Define the Laplacian kernel for averaging
    L_d = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    
    # Iteratively update u and v
    for _ in range(n_iters):
        # Compute the average of the neighboring values
        u_avg = convolve2d(u, L_d, mode='same', boundary='symm')
        v_avg = convolve2d(v, L_d, mode='same', boundary='symm')
        
        # Compute P and D
        P = I_x * u_avg + I_y * v_avg + I_t
        D = lmbd + I_x**2 + I_y**2
        
        # Update u and v
        u = u_avg - I_x * (P / D)
        v = v_avg - I_y * (P / D)
    
    return u, v