# Parallel Image Filtering Benchmark System

![Python](https://img.shields.io/badge/python-3.13+-blue.svg?logo=python&logoColor=white)
![Numba](https://img.shields.io/badge/numba-0.62.1-orange.svg)
![MPI4Py](https://img.shields.io/badge/mpi4py-4.1.1-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-NVIDIA-76B900?logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0-red?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-2.2.6-013243?logo=numpy&logoColor=white)

This project is an advanced benchmarking environment designed to analyze the performance of classic convolution algorithms (Gaussian Blur, Sobel Edge Detection, Sharpening) using various parallel and distributed programming models.



## 1. Architecture and Technologies

The system implements four primary computing paradigms:

* **Sequential (Numba JIT):** Baseline sequential implementation optimized with Just-In-Time compilation to achieve C-like performance.
* **Parallel (Numba OpenMP):** Local CPU multithreading using `prange` and `@njit(parallel=True)` for loop-level parallelization.
* **Distributed (MPI):** Distributed batch processing using `mpi4py`. Files are distributed across worker processes to minimize the total Wall-Clock Time.
* **GPU Acceleration (CUDA):** High-performance kernels implemented for NVIDIA GPUs using `numba.cuda` to leverage massively parallel hardware.



## 2. Requirements

### Python Dependencies
Install the required packages via pip:
```bash
pip install -r requirements.txt
```

### System Tools

To ensure full functionality of the benchmarking suite, the following system-level tools must be installed and configured:

* **MPI Implementation:** Required for the `mpi4py` library to facilitate inter-process communication.
    * **Windows:** [Microsoft MPI (MS-MPI)](https://learn.microsoft.com/pl-pl/message-passing-interface/microsoft-mpi) — Pobierz i zainstaluj pliki `msmpisetup.exe` oraz `msmpisdk.msi`.
    * **Linux:** [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/) (e.g., `sudo apt install libopenmpi-dev openmpi-bin`).



* **NVIDIA CUDA:** Compatible NVIDIA drivers and the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) are strictly required for executing GPU-accelerated filter variants. Ensure your GPU supports the required Compute Capability.

## 3. Usage Instructions

The program is executed through `main.py` with specific flags to determine the mode and variant of processing.

### A. Single Process Mode (Numba, CUDA)
This mode is designed for local performance testing on a single machine, utilizing either CPU multithreading or GPU acceleration.

**Command syntax:**
```bash
python main.py --mode single --variants [seq|par|cuda]
```

### B. Distributed Mode (MPI)
This mode is used for scaling tests across multiple processes, distributing the workload across CPU cores or network nodes. It requires an MPI runner such as `mpirun` or `mpiexec`.



**Command syntax:**
```bash
mpiexec -n 8 python main.py --mode mpi --variants [seq|par]
```

* `--variants seq`: **Pure MPI** – Executes one Numba thread per process. This configuration typically offers the best performance for batch processing by minimizing thread synchronization overhead and context switching between CPU cores.



* `--variants par`: **Hybrid Parallelism** – Combines MPI process distribution with Numba-based local multithreading. Each MPI process handles its assigned subset of images using all available local CPU threads via `@njit(parallel=True)`.

## 4. Benchmark Insights

The following observations are based on comprehensive tests performed on a **13th Gen Intel Core i5** CPU and an **NVIDIA RTX 3050** GPU:

* **Pure MPI Dominance:** Using **Pure MPI (8 processes)** proved to be the most efficient solution for large batch processing. In this specific scenario, it consistently outperformed the CUDA variant.



* **CUDA Bottleneck:** The primary limitation for the GPU variant in I/O intensive tasks (such as filtering hundreds of individual files) is the **Host-Device-Host** data transfer overhead. The time required to move image data to the GPU memory and back often exceeds the actual computation time for simple convolution kernels.



* **Scalability Threshold:** Optimal acceleration was observed at **8 MPI processes**. Beyond this threshold, the gains from additional parallelization are diminished by increasing communication and synchronization overhead between processes.



* **Hybrid Parallelism Performance:** Contrary to expectations, combining MPI with local Numba multithreading (Hybrid) often introduced additional thread management overhead, making it less efficient than Pure MPI for these specific workloads.

## 5. Metrics

To ensure the accuracy and reliability of the filtering process, the system automatically validates the processed images against the original input using the following quality metrics:

* **PSNR (Peak Signal-to-Noise Ratio):** Measures the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Higher values indicate better reconstruction quality.
* **SSIM (Structural Similarity Index):** A perception-based model that considers image degradation as perceived change in structural information. It ranges from -1 to 1, where 1 indicates identical images.



---

**Authors:** **Jakub Pedryc** and **Maciej Łabuz**