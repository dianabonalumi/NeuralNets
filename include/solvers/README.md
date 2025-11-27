## Build and Run

To compile the SIM code with full hardware optimizations (AVX2, FMA) and OpenBLAS support, use the following command:

```bash
g++ -std=c++17 -O3 -mavx2 -mfma -fopenmp \
    src/main.cpp \
    -o benchmark_test \
    -I./include -I/opt/OpenBLAS/include \
    -L/opt/OpenBLAS/lib -lopenblas \
    -lpthread





> *Note: Please ensure the OpenBLAS include and library paths (`-I` and `-L` flags) match the installation directory on your machine.*
