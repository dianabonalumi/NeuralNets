# Matrix Multiplication Solvers

This directory contains various C++ implementations of General Matrix Multiplication (GEMM). The solvers are designed to demonstrate the progression from a naive $O(N^3)$ implementation to a highly optimized version utilizing Cache Tiling, SIMD (AVX/AVX2), Register Blocking, and OpenMP parallelism.

All classes inherit from a common `Matrix_Solver<T>` interface.

## Optimization Strategies

The solvers are categorized by the specific optimization techniques applied:

### 1. Baseline
* **Naive Solver** (`naive_solver.hpp`)
    * Standard triple-nested loop implementation ($i, j, k$).
    * Serves as the baseline for performance comparisons.

### 2. Loop Optimizations & Caching
* **Loop Unroll Solver** (`loop_unroll_solver.hpp`)
    * Unrolls the innermost $k$-loop by a factor of 8 to reduce loop overhead and improve instruction pipelining.
* **Tiling Solver** (`tiling_solver.hpp`)
    * **Note:** Despite the name, this implements **Loop Interchange** ($i, k, j$ ordering).
    * Accesses matrix $B$ sequentially in the inner loop to improve spatial locality and reduce cache misses.

### 3. OpenMP Parallelism
* **OpenMP Solver** (`openmp_solver.hpp`)
    * Adds multi-threading to the Naive approach using `#pragma omp parallel for collapse(2)`.
* **Tiling + OpenMP Solver** (`tiling_openmp_solver.hpp`)
    * Combines Loop Interchange ($i, k, j$) with OpenMP parallelism on the outer loops.

### 4. SIMD & Vectorization (AVX/AVX2)
* **SIMD Unroll Solver** (`simd_unroll_solver.hpp`)
    * Uses **AVX Intrinsics** (`_mm256`) to process 8 floats or 4 doubles simultaneously.
    * Uses **1D Register Blocking**: Unrolls the $j$ loop (factor 8) to accumulate results into multiple vector registers (`c0`...`c7`), maximizing throughput.
* **SIMD Unroll 2D Solver** (`simd_unroll_solver_2D.hpp`)
    * Uses **2D Register Blocking**: Processes 2 rows of matrix $A$ simultaneously against blocks of matrix $B$.
    * Increases arithmetic intensity (compute-to-memory-access ratio).

### 5. Hybrid
* **All Solver** (`all_solver.hpp`)
    * Combines all previous techniques into a single solver:
        * **Explicit Block Tiling:** Blocks loops for $M, N, K$ to fit data into L1/L2 cache.
        * **OpenMP:** Parallelizes outer tiles.
        * **2D Register Blocking:** Processes 2 rows of $A$ at a time.
        * **SIMD Unrolling:** Vectorized inner loops.

## Requirements

* **Compiler:** Must support C++11 or higher.
* **Hardware:** CPU with AVX/AVX2 support (for SIMD solvers).
* **Libraries:** OpenMP (usually included with GCC/Clang/MSVC).

## Compilation Flags

To ensure intrinsics and OpenMP work correctly, compile with flags similar to:

```bash
g++ -O3 -mavx2 -mfma -fopenmp main.cpp -o matrix_test