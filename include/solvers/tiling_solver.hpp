#ifndef TILING_SOLVER_HPP
#define TILING_SOLVER_HPP

#include "../matrix_solver.hpp" // Include interface

template <typename T>
class Naive_Solver : public Matrix_Solver<T> {
public:
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {
        // Initialize C to zero
        for (int i = 0; i < M * N; ++i) C[i] = 0;
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {

                // Preload A[i][k] into a register (constant for the entire inner loop)
                T r = A[i * K + k];

                for (int j = 0; j < N; ++j) {
                    // C[i][j] += A[i][k] * B[k][j]
                    C[i * N + j] += r * B[k * N + j];
                }
            }
        }
    }

    std::string getName() const override { return "Tiling Solver"; }

};



#endif
