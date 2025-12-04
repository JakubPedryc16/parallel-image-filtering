# TYPY DYSTRYBUCJI (Argument --mode)
MODE_SINGLE = 'single'  # Zastępuje 'seq'
MODE_MPI = 'mpi'

# TYPY WARIANTÓW PRZETWARZANIA (Argument --variants)
VARIANT_SEQ = 'seq'
VARIANT_PAR = 'par'
VARIANT_CUDA = 'cuda'

VALID_MODES = [MODE_SINGLE, MODE_MPI]
VALID_VARIANTS = [VARIANT_SEQ, VARIANT_PAR, VARIANT_CUDA]

# # Tryb Pojedynczego Procesu (--mode single)
# python main.py --mode single --variants seq
# python main.py --mode single --variants par
# python main.py --mode single --variants cuda
# python main.py --mode single --variants seq,par,cuda

# # Tryb Rozproszony MPI (--mode mpi)
# mpiexec -n 4 python main.py --mode mpi --variants seq
# mpiexec -n 4 python main.py --mode mpi --variants par
# mpiexec -n 4 python main.py --mode mpi --variants seq,par