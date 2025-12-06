# How to run the testbench
1)
        export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH


2)
        g++ -std=c++17 -O3 -march=native mainT.cpp -o benchmark_test -I./include -I/opt/OpenBLAS/include -L/opt/OpenBLAS/lib -lopenblas -lpthread -fopenmp


3)
        ./benchmark_test solver_type data_type

**solver_type options:**
* naive (Naive Solver)
* simd (Solver with SIMD)
* omp (Solver with OpenMP)
* unroll (Solver with Loop Unroll)
* tiling (Solver with Tiling)
* simd_unroll_1d (Solver with a combination of SIMD and Loop Unroll 1D)
* simd_unroll_2d (Solver with a combination of SIMD and Loop Unroll 2D)
* tiling_omp (Solver with a combination of Tiling and OpenMP)
* all (Solver with a combination of Loop Unroll 2D, SIMD, Tiling and OpenMP)

**data_type options:**
* float
* double



# How to generate graphs
    python3 plot_results.py solver_type data_type