#ifndef NAIVE_SOLVER_HPP
#define NAIVE_SOLVER_HPP

#include "../matrix_solver.hpp" // Includi l'interfaccia

template <typename T>
class Naive_Solver : public Matrix_Solver<T> {
public:
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {
        // Implementazione dei 3 cicli for
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

    std::string getName() const override { return "Naive Solver"; }

};



#endif
