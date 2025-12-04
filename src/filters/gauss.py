import numpy as np
from numba import njit, prange, cuda

def make_gaussian_kernel(size=5, sigma=1.0):
    ax = np.linspace(-(size - 1)/2, (size - 1)/2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

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

@cuda.jit
def gaussian_kernel_cuda(image_gpu, output_gpu, kernel, h, w, k):
    x, y = cuda.grid(2)

    if k <= y < h - k and k <= x < w - k:
        
        val = 0.0 # Upewnij się, że suma jest inicjowana jako zmiennoprzecinkowa
        
        for ky_i in range(-k, k + 1):
            for kx_j in range(-k, k + 1):
                pixel_val = image_gpu[y + ky_i, x + kx_j]
                kernel_val = kernel[ky_i + k, kx_j + k]
                
                # To jest poprawne, bo pixel_val i kernel_val są float32
                val += pixel_val * kernel_val 
                
        output_gpu[y, x] = val


def gaussian_blur_cuda(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2

    # Upewnienie się, że obraz wejściowy jest float32
    image_gpu = cuda.to_device(image.astype(np.float32)) 
    kernel_gpu = cuda.to_device(kernel)
    output_gpu = cuda.device_array_like(image_gpu)

    threads_per_block = (32, 32)
    blocks_x = (w + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (h + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_x, blocks_y)

    gaussian_kernel_cuda[blocks_per_grid, threads_per_block](
        image_gpu, output_gpu, kernel_gpu, h, w, k
    )

    output = output_gpu.copy_to_host()
    
    # Poprawiony krok: zaokrąglenie i obcięcie do zakresu [0, 255]
    output_final = np.clip(np.round(output), 0, 255)
    
    return output_final.astype(np.uint8)