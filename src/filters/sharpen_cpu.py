import numpy as np
from numba import njit, prange

def make_sharpen_kernel():
    return np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]], dtype=np.float32)

@njit(parallel=True)
def sharpen(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2
    output = np.zeros_like(image)
    for y in prange(k, h - k):
        for x in prange(k, w - k):
            region = image[y-k:y+k+1, x-k:x+k+1]
            output[y, x] = np.sum(region * kernel)
    return output

def sharpen_seq(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2
    output = np.zeros_like(image)
    for y in range(k, h - k):
        for x in range(k, w - k):
            region = image[y-k:y+k+1, x-k:x+k+1]
            output[y, x] = np.sum(region * kernel)
    return output
