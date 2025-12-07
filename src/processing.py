import os
import cv2
import time
import numpy as np
import sys
from src.metrics import evaluate
from src.constants import VARIANT_SEQ, VARIANT_PAR, VARIANT_CUDA


try:
    from numba import cuda
except ImportError:
    class DummyCUDA:
        @staticmethod
        def is_available():
            return False
    cuda = DummyCUDA()


def run_dummy_compilation(filters_config, variants_to_run):
    
    H, W = 10, 10
    dummy_img = np.zeros((H, W), dtype=np.uint8) 
    dummy_out = np.zeros_like(dummy_img, dtype=np.uint8)
    
    for f in filters_config:
        kernel = f["make_kernel"]()
        
        if f["name"] == "edges":
            kx, ky = kernel 
            
            args_2 = (dummy_img, kx, ky)
            args_3 = (dummy_img, kx, ky, dummy_out)
        else:
            args_2 = (dummy_img, kernel)
            args_3 = (dummy_img, kernel, dummy_out)
            
        
        for variant in [VARIANT_SEQ, VARIANT_PAR]:
            if variant in variants_to_run and variant in f:
                try:
                    f[variant](*args_3) 
                except (TypeError, ValueError):
                    try:
                        f[variant](*args_2)
                    except TypeError as e:
                        print(f"BŁĄD KOMPILACJI WSTĘPNEJ dla {f['name']} ({variant}): Nieoczekiwana sygnatura funkcji.", file=sys.stderr)
                        raise e


        if VARIANT_CUDA in variants_to_run and VARIANT_CUDA in f:
            cuda_args = (dummy_img, kx, ky) if f["name"] == "edges" else (dummy_img, kernel)
            
            if 'cuda' in locals() and cuda.is_available():
                f[VARIANT_CUDA](*cuda_args)


import os
import cv2
import time
import numpy as np
from src.metrics import evaluate

def process_single_file(file_name, filter_info, input_dir, output_dirs, filter_func, method_name):
    filter_name = filter_info["name"]
    make_kernel = filter_info["make_kernel"]

    img_path = os.path.join(input_dir, file_name)
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    
    if img_color is None:
        raise Exception(f"Nie można wczytać pliku: {img_path}")
        
    img_float = img_color.astype(np.float32) 
    channels = cv2.split(img_float)
    
    kernel = make_kernel()
    is_edges = filter_name == "edges"
    results_chan = []
    
    start_wall = time.time()
    
    for channel in channels:
        
        if is_edges:
            kx, ky = kernel
            result_chan = filter_func(channel, kx, ky)
        else:
            result_chan = filter_func(channel, kernel)
            
        results_chan.append(result_chan)

    time_total = time.time() - start_wall

    result_color = cv2.merge(results_chan)

    result_color_uint8 = np.clip(result_color, 0, 255).astype(np.uint8)
    
    result_gray_uint8 = cv2.cvtColor(result_color_uint8, cv2.COLOR_BGR2GRAY) 

    original_gray_uint8 = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    psnr, ssim = evaluate(original_gray_uint8, result_gray_uint8)
    
    base_output_dir = output_dirs[filter_name]
    output_filename = f"{filter_name}_{method_name}_{file_name}"

    color_dir = os.path.join(base_output_dir, "color")
    os.makedirs(color_dir, exist_ok=True)
    cv2.imwrite(os.path.join(color_dir, output_filename), result_color_uint8)

    if is_edges:
        gray_dir = os.path.join(base_output_dir, "gray")
        os.makedirs(gray_dir, exist_ok=True)
        cv2.imwrite(os.path.join(gray_dir, output_filename), result_gray_uint8)
    
    return {
        "time": time_total,
        "psnr": psnr,
        "ssim": ssim,
        "count": 1
    }