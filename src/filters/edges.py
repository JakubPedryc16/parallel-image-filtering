import math
import numpy as np
from numba import njit, prange, cuda

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
            output[y, x] = min(max(val, 0), 255)
    return output.astype(np.uint8)

@njit
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

@cuda.jit
def sobel_kernel_cuda(image_gpu, output_gpu, kx, ky, h, w, k):
    x, y = cuda.grid(2)

    if k <= y < h - k and k <= x < w - k:
        
        gx = 0.0
        gy = 0.0
        
        for ky_i in range(-k, k + 1):
            for kx_j in range(-k, k + 1):
                pixel_val = image_gpu[y + ky_i, x + kx_j]
                
                gx += pixel_val * kx[ky_i + k, kx_j + k]
                gy += pixel_val * ky[ky_i + k, kx_j + k]
                
        val = math.sqrt(gx**2 + gy**2)
        
        output_gpu[y, x] = min(max(val, 0), 255)

def sobel_edges_cuda(image, kx, ky):
    h, w = image.shape
    k = kx.shape[0] // 2

    image_gpu = cuda.to_device(image.astype(np.float32))
    kx_gpu = cuda.to_device(kx)
    ky_gpu = cuda.to_device(ky)
    output_gpu = cuda.device_array_like(image_gpu)

    threads_per_block = (32, 32)
    
    blocks_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    sobel_kernel_cuda[blocks_per_grid, threads_per_block](
        image_gpu, output_gpu, kx_gpu, ky_gpu, h, w, k
    )

    output = output_gpu.copy_to_host()
    
    return output.astype(np.uint8)