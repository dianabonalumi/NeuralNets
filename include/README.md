# Include Directory

This directory serves as the core header repository for the Neural Network hands-on. It contains the fundamental interfaces, the solver factory, and organized subdirectories for algorithms, utilities, and application logic.

## Core Interfaces

### `matrix_solver.hpp`
Defines the abstract base class `Matrix_Solver<T>`. This interface enforces a standard structure for all matrix multiplication algorithms.
* **Virtual Method:** `multiply(int M, int N, int K, const T* A, const T* B, T* C)`.
* **Helper:** `getName()` returns a string identifier for the solver.

### `factory_m.hpp`
Implements the **Factory Design Pattern** to manage the instantiation of different solver algorithms.
* **`SolverType` Enum:** Defines available solvers (e.g., `NAIVE`, `SIMD`, `TILING`, `OPENMP`, `ALL`).
* **`SolverFactory` Class:** Provides a static `createSolver` method that returns a `std::unique_ptr` to a specific solver implementation based on the requested type.

## Directory Structure

### `solvers/`
Contains the specific header-only implementations of various GEMM (General Matrix Multiply) algorithms, including:
* **Baseline:** `naive_solver.hpp`
* **Optimized:** `simd_solver.hpp`, `loop_unroll_solver.hpp`, `tiling_solver.hpp`
* **Parallel:** `openmp_solver.hpp`, `tiling_openmp_solver.hpp`
* **Hybrid:** `all_solver.hpp` (Combines Tiling, OpenMP, and SIMD)

### `NeuralNetwork/`
Contains the implementation of the Neural Network components (e.g., Layers, Network topology) that utilize the solvers for forward and backward propagation.

### `utilities/`
Contains support files for benchmarking and analysis:
* **`benchmarkSuite`**: Logic to measure execution time and GFLOPS.
* **`mainT`**: The main testbench entry point.
* **`plot_results`**: Scripts for visualizing performance data.

### `repo_mtx/`
A repository directory used to store the generated reports and performance logs produced by the solvers during execution.