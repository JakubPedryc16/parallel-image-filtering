import numpy as np
from numba import njit, prange

def make_sobel_kernels():
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)
    return kx, ky

@njit(parallel=True)
def sobel_edges(image, kx, ky):
    h, w = image.shape
    k = kx.shape[0] // 2
    output = np.zeros_like(image, dtype=np.float32)
    for y in prange(k, h - k):
        for x in prange(k, w - k):
            region = image[y-k:y+k+1, x-k:x+k+1]
            gx = np.sum(region * kx)
            gy = np.sum(region * ky)
            val = np.sqrt(gx**2 + gy**2)
            output[y, x] = min(max(val, 0), 255)  # clip 0-255
    return output.astype(np.uint8)

def sobel_edges_seq(image, kx, ky):
    h, w = image.shape
    k = kx.shape[0] // 2
    output = np.zeros_like(image, dtype=np.float32)
    for y in range(k, h - k):
        for x in range(k, w - k):
            region = image[y-k:y+k+1, x-k:x+k+1]
            gx = np.sum(region * kx)
            gy = np.sum(region * ky)
            val = np.sqrt(gx**2 + gy**2)
            output[y, x] = min(max(val, 0), 255)
    return output.astype(np.uint8)
