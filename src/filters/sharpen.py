import numpy as np
from numba import njit, prange, cuda


def make_sharpen_kernel():
    # Kernel wyostrzający, suma wag = 1.0
    return np.array([[0, -0.25, 0],
                     [-0.25, 2, -0.25],
                     [0, -0.25, 0]], dtype=np.float32)

@njit(parallel=True)
def sharpen(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2
    output = np.zeros_like(image, dtype=np.float32)

    for y in prange(h):
        for x in prange(w):
            acc = 0.0
            for ky in range(-k, k + 1):
                for kx in range(-k, k + 1):
                    iy = min(max(y + ky, 0), h - 1)
                    ix = min(max(x + kx, 0), w - 1)
                    acc += image[iy, ix] * kernel[ky + k, kx + k]
            output[y, x] = acc

    # POPRAWKA: Obcinanie do zakresu 0.0 do 255.0
    output = np.clip(output, 0.0, 255.0)
    return output

@njit
def sharpen_seq(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2
    output = np.zeros_like(image, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            acc = 0.0
            for ky in range(-k, k + 1):
                for kx in range(-k, k + 1):
                    iy = min(max(y + ky, 0), h - 1)
                    ix = min(max(x + kx, 0), w - 1)
                    acc += image[iy, ix] * kernel[ky + k, kx + k]
            output[y, x] = acc

    # POPRAWKA: Obcinanie do zakresu 0.0 do 255.0
    output = np.clip(output, 0.0, 255.0)
    return output

@cuda.jit
def sharpen_kernel_cuda(image_gpu, output_gpu, kernel, h, w, k):
    x, y = cuda.grid(2)

    if 0 <= y < h and 0 <= x < w:
        acc = 0.0
        
        for ky_i in range(-k, k + 1):
            for kx_j in range(-k, k + 1):

                iy = min(max(y + ky_i, 0), h - 1)
                ix = min(max(x + kx_j, 0), w - 1)
                
                pixel_val = image_gpu[iy, ix]
                kernel_val = kernel[ky_i + k, kx_j + k]
                
                acc += pixel_val * kernel_val
        
        # POPRAWKA: Obcinanie do zakresu 0.0 do 255.0 w jądrze CUDA
        output_gpu[y, x] = max(0.0, min(255.0, acc)) 

def sharpen_cuda(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2
    
    # Zakładamy, że obraz wejściowy jest już w float32
    image_gpu = cuda.to_device(image.astype(np.float32)) 
    kernel_gpu = cuda.to_device(kernel)
    output_gpu = cuda.device_array_like(image_gpu)

    threads_per_block = (32, 32)
    
    blocks_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    sharpen_kernel_cuda[blocks_per_grid, threads_per_block](
        image_gpu, output_gpu, kernel_gpu, h, w, k
    )

    output = output_gpu.copy_to_host()
    
    return output