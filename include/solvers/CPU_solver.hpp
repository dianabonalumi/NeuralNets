#ifndef CPUSOLVER_HPP
#define CPUSOLVER_HPP

#include "Matrix_Solver.hpp"

template <typename T>
class CPUSolver : public Matrix_Solver<T> {
public:
    // Implementazione della moltiplicazione righe-per-colonne
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {
        // C = A * B
        // A è (M x K), B è (K x N), C è (M x N)
        
        // 1. Pulisci la memoria di destinazione (importante!)
        for (int i = 0; i < M * N; ++i) {
            C[i] = 0;
        }

        // 2. Triplo loop standard
        for (int i = 0; i < M; ++i) {        // Per ogni riga di A
            for (int j = 0; j < N; ++j) {    // Per ogni colonna di B
                T sum = 0;
                for (int p = 0; p < K; ++p) { // Prodotto scalare
                    sum += A[i * K + p] * B[p * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    std::string getName() const override { return "CPU Solver (Naive)"; }
};

#endif