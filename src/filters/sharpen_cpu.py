import numpy as np
from numba import njit, prange

def make_sharpen_kernel():
    # mniej agresywny kernel, daje naturalniejsze wyostrzenie
    return np.array([[0, -0.25, 0],
                     [-0.25, 2, -0.25],
                     [0, -0.25, 0]], dtype=np.float32)


@njit(parallel=True)
def sharpen(image, kernel):
    h, w = image.shape
    k = kernel.shape[0] // 2
    output = np.zeros_like(image, dtype=np.float32)

    # iterujemy tylko po bezpiecznym obszarze, a na brzegach kopiujemy wartości
    for y in prange(h):
        for x in prange(w):
            acc = 0.0
            for ky in range(-k, k + 1):
                for kx in range(-k, k + 1):
                    iy = min(max(y + ky, 0), h - 1)
                    ix = min(max(x + kx, 0), w - 1)
                    acc += image[iy, ix] * kernel[ky + k, kx + k]
            output[y, x] = acc

    # przycinanie do zakresu [0, 1]
    output = np.clip(output, 0.0, 1.0)
    return output


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

    output = np.clip(output, 0.0, 1.0)
    return output
