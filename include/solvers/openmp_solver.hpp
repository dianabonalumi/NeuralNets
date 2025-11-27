#ifndef OMP_SOLVER_HPP
#define OMP_SOLVER_HPP

#include <omp.h>
#include "../matrix_solver.hpp"

template <typename T>
class OpenMP_Solver : public Matrix_Solver<T> {
public:
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {
        

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T sum = 0;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    std::string getName() const override { return "OpenMP Solver"; }
};

#endif
