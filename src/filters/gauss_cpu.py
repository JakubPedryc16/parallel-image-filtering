
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def gaussian_blur(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2
    output = np.zeros_like(image)
    for y in prange(k, h - k):
        for x in prange(k, w - k):
            region = image[y-k:y+k+1, x-k:x+k+1]
            output[y, x] = np.sum(region * kernel)
    return output

def gaussian_blur_seq(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2
    output = np.zeros_like(image)
    for y in range(k, h - k):
        for x in range(k, w - k):
            region = image[y-k:y+k+1, x-k:x+k+1]
            output[y, x] = np.sum(region * kernel)
    return output


def make_gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size - 1)/2, (size - 1)/2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)
