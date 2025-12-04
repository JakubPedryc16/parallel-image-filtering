import csv
import json
import os
from src.constants import VARIANT_SEQ, VARIANT_PAR, VARIANT_CUDA

def get_variant_description(variant_name):
    if variant_name == VARIANT_SEQ:
        return "Bazowy/Sekwencyjny (Akceleracja niejawna CPU)"
    elif variant_name == VARIANT_PAR:
        return "Rownolegle (Numba/OpenMP)"
    elif variant_name == VARIANT_CUDA:
        return "Akceleracja GPU (Numba CUDA)"
    elif variant_name == 'mpi': 
        return "Rozproszony (MPI)" 
    
    else:
        return "Nieznany Wariant"

def calculate_averages(totals):
    count = totals.get("count", 0)
    if count == 0:
        return 0.0, 0.0, 0.0, 0
    
    avg_time = totals["time"] / count
    avg_psnr = totals["psnr"] / count
    avg_ssim = totals["ssim"] / count
    return avg_time, avg_psnr, avg_ssim, count


def display_mpi_results(filters_config, global_results):
    for f_info in filters_config:
        filter_name = f_info["name"]
        
        print(f"\n--- {filter_name.capitalize()} - Benchmark Dystrybuowany (MPI) ---")
        
        seq_key = f"{filter_name}_{VARIANT_SEQ}"
        mpi_key = f"{filter_name}_mpi" 
        
        keys_to_check = [
            seq_key, 
            mpi_key 
        ]
        
        ref_speedup_res = None
        any_variant_processed = False 
        
        for key in keys_to_check:
            if key in global_results and global_results[key]['count'] > 0:
                
                any_variant_processed = True 
                res = global_results[key]
                
                if ref_speedup_res is None:
                    ref_speedup_res = {'wall_time': res['wall_time'], 'key': key, 'count': res['count']}
                
                variant_name = key.split('_')[-1] 
                
                desc_full = get_variant_description(variant_name) 
                
                print(
                    f"{desc_full.upper()} -> "
                    f"czas/plik: {res['avg_time']:.4f}s | "
                    f"PSNR: {res['avg_psnr']:.2f} dB, SSIM: {res['avg_ssim']:.4f}"
                )
                
                print(f"   -> CALKOWITY CZAS (Wall-Clock): {res['wall_time']:.4f}s")

        if ref_speedup_res and ref_speedup_res['count'] > 0:
            
            ref_wall_time = ref_speedup_res['wall_time']
            ref_name_full = ref_speedup_res['key'].split('_')[-1].upper()
            
            print(f"\n--- Analiza Przyspieszenia (Speedup vs. {ref_name_full}: {ref_wall_time:.4f}s) ---") 

            for key in keys_to_check:
                if key in global_results and global_results[key]['count'] > 0:
                    current_res = global_results[key]
                    
                    if current_res['wall_time'] != ref_wall_time:
                        speedup = ref_wall_time / current_res['wall_time']
                        variant_name = key.split('_')[-1].upper()
                        print(f"  -> Przyspieszenie {variant_name.upper()}: {speedup:.2f}x")
                    else:
                        variant_name = key.split('_')[-1].upper()
                        print(f"  -> {variant_name.upper()} jest punktem odniesienia: 1.00x")
            
        elif not any_variant_processed:
            print(f" Brak danych do wyswietlenia dla filtru {filter_name}.")

def display_sequential_results(filters_config, global_results):
    for f_info in filters_config:
        filter_name = f_info["name"]
        
        print(f"\n--- {filter_name.capitalize()} - Standardowy Benchmark ---")
        
        seq_key = f"{filter_name}_{VARIANT_SEQ}"
        
        keys_to_check = [
            seq_key, 
            f"{filter_name}_{VARIANT_PAR}", 
            f"{filter_name}_{VARIANT_CUDA}"
        ]
        
        ref_speedup_res = None
        any_variant_processed = False 
        
        for key in keys_to_check:
            if key in global_results and global_results[key]['count'] > 0:
                
                any_variant_processed = True 
                res = global_results[key]
                
                if ref_speedup_res is None:
                    ref_speedup_res = {'wall_time': res['wall_time'], 'key': key, 'count': res['count']}
                
                variant_name = key.split('_')[-1] 
                
                desc_full = get_variant_description(variant_name)
                desc_short = {
                    VARIANT_SEQ: ' 1.',
                    VARIANT_PAR: ' 2.',
                    VARIANT_CUDA: ' 4.'
                }.get(variant_name, ' X.')

                print(
                    f"{desc_short} {desc_full} -> "
                    f"czas/plik: {res['avg_time']:.4f}s | "
                    f"PSNR: {res['avg_psnr']:.2f} dB, SSIM: {res['avg_ssim']:.4f}"
                )
                
                print(f"   -> CALKOWITY CZAS (Wall-Clock): {res['wall_time']:.4f}s")

        if ref_speedup_res and ref_speedup_res['count'] > 0:
            
            ref_wall_time = ref_speedup_res['wall_time']
            ref_name_full = ref_speedup_res['key'].split('_')[-1].upper()
            
            print(f"\n--- Analiza Przyspieszenia (Speedup vs. {ref_name_full}: {ref_wall_time:.4f}s) ---") 

            for key in keys_to_check:
                if key in global_results and global_results[key]['count'] > 0:
                    current_res = global_results[key]
                    
                    if current_res['wall_time'] != ref_wall_time:
                        speedup = ref_wall_time / current_res['wall_time']
                        variant_name = key.split('_')[-1].upper()
                        print(f"  -> Przyspieszenie {variant_name}: {speedup:.2f}x")
                    else:
                        variant_name = key.split('_')[-1].upper()
                        print(f"  -> {variant_name} jest punktem odniesienia: 1.00x")
            
        elif not any_variant_processed:
            print(f" Brak danych do wyswietlenia dla filtru {filter_name}.")


import os
import json
import csv

def save_results_to_file(global_results, output_filename, mode, filters_config=None):
    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"{output_filename}.csv")
    csv_data = []

    fieldnames = [
        "filter", "variant", "mode", "avg_time_s", "wall_time_s", "psnr_db", "ssim", "count"
    ]

    for key, res in global_results.items():
        filter_name = key.split('_')[0]
        variant_name = key.split('_')[-1]
        
        csv_data.append({
            "filter": filter_name,
            "variant": variant_name,
            "mode": mode,
            "avg_time_s": f"{res['avg_time']:.4f}",
            "wall_time_s": f"{res.get('wall_time', 'N/A'):.4f}", 
            "psnr_db": f"{res['avg_psnr']:.2f}",
            "ssim": f"{res['avg_ssim']:.4f}",
            "count": res['count']
        })

    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\n[OK] Zapisano wyniki benchmarku do: {csv_path}")
    except Exception as e:
        print(f"\n[ERROR] BLAD zapisu CSV: {e}")

    json_path = os.path.join(output_dir, f"{output_filename}.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(global_results, jsonfile, indent=4)
        print(f"[OK] Zapisano pelne dane do: {json_path}")
    except Exception as e:
        print(f"[ERROR] BLAD zapisu JSON: {e}")