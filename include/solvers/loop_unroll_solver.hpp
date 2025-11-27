#ifndef LOOP_UNROLL_SOLVER_HPP
#define LOOP_UNROLL_SOLVER_HPP

#include "../matrix_solver.hpp" // Include interface

template <typename T>
class Loop_Unroll_Solver : public Matrix_Solver<T> {
public:
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T sum = 0;
                int k = 0;

                // Unrolled cycle (factor 8)
                for (; k <= K - 8; k += 8) {
                    sum += A[i * K + k]     * B[k * N + j];
                    sum += A[i * K + k + 1] * B[(k + 1) * N + j];
                    sum += A[i * K + k + 2] * B[(k + 2) * N + j];
                    sum += A[i * K + k + 3] * B[(k + 3) * N + j];
                    sum += A[i * K + k + 4] * B[(k + 4) * N + j];
                    sum += A[i * K + k + 5] * B[(k + 5) * N + j];
                    sum += A[i * K + k + 6] * B[(k + 6) * N + j];
                    sum += A[i * K + k + 7] * B[(k + 7) * N + j];
                }

                // Cleanup
                for (; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }

                C[i * N + j] = sum;
            }
        }
    }

    std::string getName() const override { return "Loop Unroll Solver"; }

};



#endif
