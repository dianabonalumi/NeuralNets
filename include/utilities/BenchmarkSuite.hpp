#ifndef BENCHMARK_SUITE_HPP
#define BENCHMARK_SUITE_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cblas.h> 
#include "../factory_m.hpp" 

// --- WRAPPERS OPENBLAS ---
template <typename T>
void call_openblas(int M, int N, int K, const T* A, const T* B, T* C);

template <>
void call_openblas<float>(int M, int N, int K, const float* A, const float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

template <>
void call_openblas<double>(int M, int N, int K, const double* A, const double* B, double* C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1.0, A, K, B, N, 0.0, C, N);
}

template <typename T>
class BenchmarkSuite {
private:
    void randomInit(std::vector<T>& vec) {
        for (auto& v : vec) {
            v = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }
    }

    bool verify(int size, const T* expected, const T* actual) {
        double epsilon = 1e-2; 
        for (int i = 0; i < size; ++i) {
            double diff = std::abs(expected[i] - actual[i]);
            if (diff > epsilon) {
                double rel_err = diff / (std::abs(expected[i]) + 1e-9);
                if (rel_err < 0.01) continue; 

                std::cerr << "ERROR at index " << i
                          << ": Expected " << expected[i]
                          << ", Obtained " << actual[i] << "\n";
                return false;
            }
        }
        return true;
    }

public:

    void checkCorrectness(int N, SolverType type) {
        int M = N;
        int K = N;
        std::vector<T> A(M * K);
        std::vector<T> B(K * N);
        std::vector<T> C_mine(M * N, 0);
        std::vector<T> C_blas(M * N, 0);

        randomInit(A);
        randomInit(B);

        call_openblas<T>(M, N, K, A.data(), B.data(), C_blas.data());

        auto solver = SolverFactory<T>::createSolver(type);
        solver->multiply(M, N, K, A.data(), B.data(), C_mine.data());

        if (!verify(M * N, C_blas.data(), C_mine.data())) {
            std::cerr << "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
            std::cerr << "CRITICAL ERROR: Solver " << solver->getName() << " gives wrong output!\n";
            std::cerr << "Interruption execution to avoid wrong benchmarks.\n";
            std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
            exit(1); 
        }
    }

    void runScalabilityTest(int max_n, SolverType type) {
        std::cout << "Size,Time_Mine,GFLOPs_Mine,Time_Blas,GFLOPs_Blas\n"; 

        for (int n = 128; n <= max_n; n += 128) {
            
            std::vector<T> A(n * n);
            std::vector<T> B(n * n);
            std::vector<T> C_mine(n * n);
            std::vector<T> C_blas(n * n);
            randomInit(A); randomInit(B);

            // OpenBLAS
            auto start_blas = std::chrono::high_resolution_clock::now();
            call_openblas<T>(n, n, n, A.data(), B.data(), C_blas.data());
            auto end_blas = std::chrono::high_resolution_clock::now();
            double duration_blas = std::chrono::duration<double>(end_blas - start_blas).count();
            double ms_blas = duration_blas * 1000.0;
            double gflops_blas = (2.0 * std::pow(n, 3)) / (duration_blas * 1e9);

            // Tuo Solver
            auto solver = SolverFactory<T>::createSolver(type);
            auto start_mine = std::chrono::high_resolution_clock::now();
            solver->multiply(n, n, n, A.data(), B.data(), C_mine.data());
            auto end_mine = std::chrono::high_resolution_clock::now();
            
            double duration_mine = std::chrono::duration<double>(end_mine - start_mine).count();
            double ms_mine = duration_mine * 1000.0;
            double gflops_mine = (2.0 * std::pow(n, 3)) / (duration_mine * 1e9);

            std::cout << n << "," 
                      << ms_mine << "," << gflops_mine << ","
                      << ms_blas << "," << gflops_blas << "\n";
        }
    }
};

#endif
