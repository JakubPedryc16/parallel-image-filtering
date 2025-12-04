import os
import time
import sys
from src.processing import process_single_file, run_dummy_compilation
from src.constants import VARIANT_SEQ, VARIANT_PAR, VARIANT_CUDA, VALID_VARIANTS, MODE_SINGLE
from src.utils import calculate_averages, get_variant_description, display_sequential_results, save_results_to_file

def process_variant_filter_combination(all_files, f_info, method_name, input_dir, output_dirs):
    filter_name = f_info["name"]
    filter_func = f_info[method_name]
    
    # Tworzenie nazwy metody, która uwzględnia tryb pracy (single)
    # Ta nazwa będzie przekazana do process_single_file i użyta do nazwania pliku wynikowego
    output_method_name = f"{MODE_SINGLE}_{method_name}"
    
    totals = {
        "time": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "count": 0
    }

    start_wall_time = time.time() 

    for file in all_files:
        try:
            # Używamy output_method_name zamiast surowego method_name
            result = process_single_file(file, f_info, input_dir, output_dirs, filter_func, output_method_name)
            
            totals["time"] += result["time"]
            totals["psnr"] += result["psnr"]
            totals["ssim"] += result["ssim"]
            totals["count"] += result["count"]
            
        except Exception as e:
            print(f"BLAD PRZETWARZANIA dla filtru {filter_name} ({output_method_name}) pliku {file}: {e}", file=sys.stderr)
            if method_name == VARIANT_CUDA:
                raise e
                
    end_wall_time = time.time()
    single_process_wall_time = end_wall_time - start_wall_time

    avg_time, avg_psnr, avg_ssim, count = calculate_averages(totals)
    
    # Używamy surowego method_name do tworzenia klucza global_results
    key = f"{filter_name}_{method_name}"
    return key, {
        "avg_time": avg_time,
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "wall_time": single_process_wall_time,
        "count": count
    }


def run_sequential_benchmark(input_dir, output_dirs, filters_config, variants_to_run):
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    run_dummy_compilation(filters_config, variants_to_run)
    
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png"))])

    if not all_files:
        print("Brak plikow do przetworzenia. Koniec dzialania.")
        sys.exit(0)

    global_results = {} 

    for f_info in filters_config:
        for method_name in variants_to_run:
            
            if method_name not in VALID_VARIANTS or method_name not in f_info:
                continue

            key, results = process_variant_filter_combination(
                all_files, f_info, method_name, input_dir, output_dirs
            )
            global_results[key] = results

    # Użycie wydzielonej funkcji wyświetlania
    display_sequential_results(filters_config, global_results)

    output_filename = f"benchmark_{MODE_SINGLE}_{'-'.join(variants_to_run)}"
    save_results_to_file(global_results, output_filename, MODE_SINGLE, filters_config)