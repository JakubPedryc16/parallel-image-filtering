import os
import numba
import numpy as np
import argparse 
import sys

from src.runner import run_sequential_benchmark
from src.constants import MODE_SINGLE, MODE_MPI, VALID_VARIANTS, VARIANT_SEQ, VARIANT_PAR, VARIANT_CUDA

try:
    from mpi4py import MPI
    from src.mpi_runner import run_mpi_benchmark
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


from src.filters.sharpen import make_sharpen_kernel, sharpen, sharpen_seq, sharpen_cuda
from src.filters.gauss import gaussian_blur, gaussian_blur_seq, make_gaussian_kernel, gaussian_blur_cuda
from src.filters.edges import sobel_edges, sobel_edges_seq, make_sobel_kernels, sobel_edges_cuda

numba.set_num_threads(1)
INPUT_DIR = "data/input"
OUTPUT_DIRS = {
    "blur": "data/output_blur",
    "edges": "data/output_edges",
    "sharpen": "data/output_sharpen"
}

USE_NUMBA_PARALLEL = True 

DEFAULT_VARIANTS = f"{VARIANT_SEQ},{VARIANT_PAR},{VARIANT_CUDA}" 

def setup_filters(numba_flag):
    
    par_gauss = gaussian_blur if numba_flag else gaussian_blur_seq
    par_sharpen = sharpen if numba_flag else sharpen_seq
    par_sobel = sobel_edges if numba_flag else sobel_edges_seq
    
    return [
        {
            "name": "blur", 
            VARIANT_SEQ: gaussian_blur_seq, 
            VARIANT_PAR: par_gauss, 
            VARIANT_CUDA: gaussian_blur_cuda, 
            "make_kernel": lambda: make_gaussian_kernel()
        },
        {
            "name": "edges", 
            VARIANT_SEQ: sobel_edges_seq, 
            VARIANT_PAR: par_sobel, 
            VARIANT_CUDA: sobel_edges_cuda, 
            "make_kernel": lambda: make_sobel_kernels()
        },
        {
            "name": "sharpen", 
            VARIANT_SEQ: sharpen_seq, 
            VARIANT_PAR: par_sharpen, 
            VARIANT_CUDA: sharpen_cuda, 
            "make_kernel": lambda: make_sharpen_kernel()
        }
    ]

FILTERS = setup_filters(USE_NUMBA_PARALLEL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uruchomienie benchmarków filtrów obrazu.")
    
    parser.add_argument("--mode", type=str, choices=[MODE_SINGLE, MODE_MPI], default=MODE_SINGLE,
                        help=f"Tryb uruchomienia: '{MODE_SINGLE}' (standardowy) lub '{MODE_MPI}' (rozproszony).")
    
    parser.add_argument("--variants", type=str, default=DEFAULT_VARIANTS,
                        help=f"Warianty do uruchomienia ({', '.join(VALID_VARIANTS)}). Domyślnie: {DEFAULT_VARIANTS}.")
    
    args = parser.parse_args()

    mode = args.mode
    variants_to_run = [v.strip() for v in args.variants.split(',') if v.strip() and v.strip() in VALID_VARIANTS]

    if not variants_to_run:
        print(f"BŁĄD: Nie wybrano poprawnych wariantów do uruchomienia. Oczekiwano: {', '.join(VALID_VARIANTS)}.")
        sys.exit(1)

    if mode == MODE_SINGLE:
        
        print("--- Uruchamiam standardowy benchmark (Tryb Pojedynczego Procesu) ---")

        run_sequential_benchmark(INPUT_DIR, OUTPUT_DIRS, FILTERS, variants_to_run)

    elif mode == MODE_MPI:
        if not MPI_AVAILABLE:
            print("BŁĄD: Wybrano tryb MPI, ale 'mpi4py' nie jest dostępna.")
            sys.exit(1)
            
        valid_mpi_variants = [VARIANT_SEQ, VARIANT_PAR]
        variants_to_test = [v for v in variants_to_run if v in valid_mpi_variants]
        
        if not variants_to_test:
            print(f"BŁĄD: Nie wybrano poprawnych wariantów do uruchomienia w trybie MPI. Oczekiwano: {', '.join(valid_mpi_variants)}.")
            sys.exit(1)

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        
        if size <= 1:
            if rank == 0:
                 print(f"BŁĄD: Wybrano tryb MPI, ale uruchomiono go jako pojedynczy proces. Użyj komendy: 'mpiexec -n X python main.py --mode {MODE_MPI} --variants <wariant>' (gdzie X > 1)")
            sys.exit(1)

        print(f"--- Uruchamiam benchmark MPI (Tryb Rozproszony) na {size} procesach ---")
        
        for mpi_variant in variants_to_test:
            if rank == 0:
                print(f"\n--- URUCHAMIANIE WARIANTU MPI: {mpi_variant.upper()} ---")

            comm.Barrier() 
            
            run_mpi_benchmark(INPUT_DIR, OUTPUT_DIRS, FILTERS, mpi_variant)
            
        comm.Barrier()