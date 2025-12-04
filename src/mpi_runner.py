from mpi4py import MPI
import os
import sys
import time
import numpy as np

from main import MPI_AVAILABLE
from src.processing import process_single_file, run_dummy_compilation
from src.constants import VARIANT_SEQ, VARIANT_PAR, VARIANT_CUDA, MODE_MPI 
from src.utils import calculate_averages, get_variant_description, display_mpi_results, save_results_to_file

try:
    from numba import cuda
except ImportError:
    pass

def handle_cuda_setup(comm, rank, variant_name):
    if variant_name != VARIANT_CUDA:
        return

    if rank == 0:
        print("BLAD KONFIGURACJI: Wykryto niestabilna kombinacje MPI + CUDA.")
        print("Wariant 'cuda' jest niedozwolony w trybie rozproszonym (MPI). PROSZE UZYC: --variants par.")
    comm.Abort(1)

def distribute_files(comm, rank, size, input_dir):
    if rank == 0:
        all_files = sorted([f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png"))])
        num_files = len(all_files)
        
        if num_files == 0:
            print("Brak plikow do przetworzenia.")
            sys.exit(0)
            
        avg = num_files // size
        remainder = num_files % size
        
        counts = [avg + 1 if i < remainder else avg for i in range(size)]
        displs = [sum(counts[:i]) for i in range(size)]
        
        file_chunks = [all_files[displs[i]:displs[i]+counts[i]] for i in range(size)]
        
        my_files = file_chunks[0]
        
        for i in range(1, size):
            comm.send(file_chunks[i], dest=i, tag=11)
            
        return my_files
    else:
        return comm.recv(source=0, tag=11)

def process_and_aggregate_filter(comm, rank, f_info, variant_name, input_dir, output_dirs, my_files):
    filter_name = f_info["name"]
    filter_func = f_info[variant_name]
    
    start_filter_time = time.time()
    
    local_totals = {
        "time": 0.0, 
        "psnr": 0.0, 
        "ssim": 0.0, 
        "count": 0
    }

    output_method_name = f"{MODE_MPI}_{variant_name}"

    for file in my_files:
        try:
            result = process_single_file(file, f_info, input_dir, output_dirs, filter_func, output_method_name)
            
            local_totals["time"] += result["time"]
            local_totals["psnr"] += result["psnr"]
            local_totals["ssim"] += result["ssim"]
            local_totals["count"] += result["count"]
            
        except Exception as e:
            print(f"BLAD PRZETWARZANIA! Proces {rank} (Wariant: {output_method_name}) napotkal blad: {e}", file=sys.stderr)
            
    global_totals = {}
    for key in local_totals:
        global_totals[key] = comm.reduce(local_totals[key], op=MPI.SUM, root=0)

    end_filter_time = time.time()
    
    return global_totals, end_filter_time - start_filter_time

def run_mpi_benchmark(input_dir, output_dirs, filters_config, variant_name):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    handle_cuda_setup(comm, rank, variant_name)

    if rank == 0:
        for dir_path in output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        global_mpi_results = {}
        
    my_files = distribute_files(comm, rank, size, input_dir)
    
    comm.Barrier()
    
    if rank == 0:
        print(f"Rank 0: Uruchamiam wstepna kompilacje wszystkich jader dla wariantu {variant_name}.")
        run_dummy_compilation(filters_config, [variant_name]) 
    
    comm.Barrier()

    VALID_MPI_VARIANTS = [VARIANT_SEQ, VARIANT_PAR]
    if variant_name not in VALID_MPI_VARIANTS:
        if rank == 0:
            print(f"BLAD: Nieznany wariant do uruchomienia wewnetrznie w MPI: {variant_name}. Oczekiwano: {', '.join(VALID_MPI_VARIANTS)}.")
        comm.Abort(1) 

    for f_info in filters_config:
        filter_name = f_info["name"]
        
        if variant_name not in f_info:
            if rank == 0:
                print(f"BLAD: Wariant '{variant_name}' nie jest zaimplementowany dla filtru '{filter_name}'.")
            comm.Abort(1)

        global_totals, mpi_wall_time = process_and_aggregate_filter(
            comm, rank, f_info, variant_name, input_dir, output_dirs, my_files
        )

        if rank == 0 and global_totals:
            
            avg_time, avg_psnr, avg_ssim, count = calculate_averages(global_totals)
            
            result_entry = {
                "avg_time": avg_time,
                "avg_psnr": avg_psnr,
                "avg_ssim": avg_ssim,
                "count": count,
                "wall_time": mpi_wall_time
            }
          
            mpi_key = f"{filter_name}_{variant_name}" 
            global_mpi_results[mpi_key] = result_entry

    comm.Barrier()
    
    if rank == 0:
        display_mpi_results(filters_config, global_mpi_results)

        output_filename = f"benchmark_{MODE_MPI}_{variant_name}_p{size}"
        save_results_to_file(global_mpi_results, output_filename, MODE_MPI)