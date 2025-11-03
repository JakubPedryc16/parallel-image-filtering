import os
import cv2
import time
import numpy as np
from src.filters.sharpen_cpu import sharpen, sharpen_seq
from src.filters.gauss_cpu import gaussian_blur, gaussian_blur_seq, make_gaussian_kernel
from src.filters.edges_cpu import sobel_edges, sobel_edges_seq, make_sobel_kernels
from src.metrics import evaluate

input_dir = "data/input"
output_dirs = {
    "blur": "data/output_blur",
    "edges": "data/output_edges",
    "sharpen": "data/output_sharpen"
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

filters = [
    {"name": "blur", "seq": gaussian_blur_seq, "par": gaussian_blur, "make_kernel": lambda: make_gaussian_kernel(5)},
    {"name": "edges", "seq": sobel_edges_seq, "par": sobel_edges, "make_kernel": make_sobel_kernels},
    {"name": "sharpen", "seq": sharpen_seq, "par": sharpen, "make_kernel": lambda: np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)}
]

dummy_img = np.zeros((10, 10), dtype=np.float32)
for f in filters:
    kernel = f["make_kernel"]()
    if f["name"] == "edges":
        kx, ky = kernel
        f["seq"](dummy_img, kx, ky)
        f["par"](dummy_img, kx, ky)
    else:
        f["seq"](dummy_img, kernel)
        f["par"](dummy_img, kernel)

for f in filters:
    total_time_seq = 0
    total_time_par = 0
    total_psnr_seq = 0
    total_psnr_par = 0
    total_ssim_seq = 0
    total_ssim_par = 0
    count = 0

    for file in sorted(os.listdir(input_dir)):
        if not file.endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        kernel = f["make_kernel"]()

        if f["name"] == "edges":
            kx, ky = kernel

            start_seq = time.time()
            result_seq = f["seq"](img, kx, ky)
            time_seq = time.time() - start_seq

            start_par = time.time()
            result_par = f["par"](img, kx, ky)
            time_par = time.time() - start_par
        else:
            start_seq = time.time()
            result_seq = f["seq"](img, kernel)
            time_seq = time.time() - start_seq

            start_par = time.time()
            result_par = f["par"](img, kernel)
            time_par = time.time() - start_par

        psnr_seq, ssim_seq = evaluate(img, result_seq)
        psnr_par, ssim_par = evaluate(img, result_par)

        total_time_seq += time_seq
        total_time_par += time_par
        total_psnr_seq += psnr_seq
        total_psnr_par += psnr_par
        total_ssim_seq += ssim_seq
        total_ssim_par += ssim_par
        count += 1

        cv2.imwrite(os.path.join(output_dirs[f["name"]], f"{f['name']}_seq_{file}"), (result_seq * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dirs[f["name"]], f"{f['name']}_par_{file}"), (result_par * 255).astype(np.uint8))

    print(f"{f['name'].capitalize()} - średnio dla {count} plików:")
    print(f"  Sekwencyjnie → czas: {total_time_seq/count:.4f}s, PSNR: {total_psnr_seq/count:.2f} dB, SSIM: {total_ssim_seq/count:.4f}")
    print(f"  Równolegle  → czas: {total_time_par/count:.4f}s, PSNR: {total_psnr_par/count:.2f} dB, SSIM: {total_ssim_par/count:.4f}")
