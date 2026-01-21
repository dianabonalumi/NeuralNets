#ifndef OMP_SOLVER_HPP
#define OMP_SOLVER_HPP

#include <omp.h>
#include "../matrix_solver.hpp"

template <typename T>
class OpenMP_Solver : public Matrix_Solver<T> {
public:
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {
        
        
        #pragma omp parallel for
        for (int i = 0; i < M * N; ++i) {
            C[i] = 0;
        }

        
        #pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                
                T a_val = A[i * K + k]; 

               
                #pragma omp simd
                for (int j = 0; j < N; ++j) {
                    C[i * N + j] += a_val * B[k * N + j];
                }
            }
        }
    }
    std::string getName() const override { return "OpenMP Solver"; }

};

#endif