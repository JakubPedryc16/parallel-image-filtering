import os
import cv2
import time

import numpy as np
from filters.gauss_cpu import gaussian_blur, gaussian_blur_seq, make_gaussian_kernel
from src.metrics import evaluate

input_dir = "data/input"
output_dir = "data/output"

kernel = make_gaussian_kernel(5, 1.2)
_ = gaussian_blur(np.zeros((10, 10), dtype=np.float32), kernel)


results = []

for file in os.listdir(input_dir):
    if file.endswith((".jpg", ".png")):
        img_path = os.path.join(input_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        start_seq = time.time()
        blurred_seq = gaussian_blur_seq(img, kernel)
        time_seq = time.time() - start_seq
        psnr_seq, ssim_seq = evaluate(img, blurred_seq)
        cv2.imwrite(os.path.join(output_dir, f"blurred_seq_{file}"), (blurred_seq * 255).astype(np.uint8))

        start_par = time.time()
        blurred_par = gaussian_blur(img, kernel)
        time_par = time.time() - start_par
        psnr_par, ssim_par = evaluate(img, blurred_par)
        cv2.imwrite(os.path.join(output_dir, f"blurred_par_{file}"), (blurred_par * 255).astype(np.uint8))

        results.append({
            "file": file,
            "time_seq": time_seq,
            "psnr_seq": psnr_seq,
            "ssim_seq": ssim_seq,
            "time_par": time_par,
            "psnr_par": psnr_par,
            "ssim_par": ssim_par
        })

        print(f"{file}:")
        print(f"  Sekwencyjnie → czas: {time_seq:.4f}s, PSNR: {psnr_seq:.2f} dB, SSIM: {ssim_seq:.4f}")
        print(f"  Równolegle  → czas: {time_par:.4f}s, PSNR: {psnr_par:.2f} dB, SSIM: {ssim_par:.4f}")

avg_time_seq = sum(r["time_seq"] for r in results) / len(results)
avg_time_par = sum(r["time_par"] for r in results) / len(results)
print(f"\nŚredni czas sekwencyjnie: {avg_time_seq:.4f}s")
print(f"Średni czas równolegle: {avg_time_par:.4f}s")
print(f"Przyspieszenie: {avg_time_seq/avg_time_par:.2f}×")
