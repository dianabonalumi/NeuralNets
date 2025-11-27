#ifndef SIMD_UNROLL_SOLVER_HPP
#define SIMD_UNROLL_SOLVER_HPP

#include <immintrin.h>
#include "../matrix_solver.hpp"

// Classe generica vuota
template <typename T>
class Simd_Unroll_Solver : public Matrix_Solver<T> {
public:
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {}
    std::string getName() const override { return "SIMD + Unroll Solver"; }
};


template <>
class Simd_Unroll_Solver<float> : public Matrix_Solver<float> {
public:
    void multiply(int M, int N, int K, const float* A, const float* B, float* C) override {
        
        for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                
                float a_val = A[i * K + k];
                __m256 a_vec = _mm256_set1_ps(a_val);

                int j = 0;

                for (; j <= N - 64; j += 64) {
                    
                    __m256 c0 = _mm256_loadu_ps(&C[i * N + j + 0]);
                    __m256 c1 = _mm256_loadu_ps(&C[i * N + j + 8]);
                    __m256 c2 = _mm256_loadu_ps(&C[i * N + j + 16]);
                    __m256 c3 = _mm256_loadu_ps(&C[i * N + j + 24]);
                    __m256 c4 = _mm256_loadu_ps(&C[i * N + j + 32]);
                    __m256 c5 = _mm256_loadu_ps(&C[i * N + j + 40]);
                    __m256 c6 = _mm256_loadu_ps(&C[i * N + j + 48]);
                    __m256 c7 = _mm256_loadu_ps(&C[i * N + j + 56]);

                    c0 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j + 0]), c0);
                    c1 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j + 8]), c1);
                    c2 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j + 16]), c2);
                    c3 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j + 24]), c3);
                    c4 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j + 32]), c4);
                    c5 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j + 40]), c5);
                    c6 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j + 48]), c6);
                    c7 = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j + 56]), c7);

                    _mm256_storeu_ps(&C[i * N + j + 0], c0);
                    _mm256_storeu_ps(&C[i * N + j + 8], c1);
                    _mm256_storeu_ps(&C[i * N + j + 16], c2);
                    _mm256_storeu_ps(&C[i * N + j + 24], c3);
                    _mm256_storeu_ps(&C[i * N + j + 32], c4);
                    _mm256_storeu_ps(&C[i * N + j + 40], c5);
                    _mm256_storeu_ps(&C[i * N + j + 48], c6);
                    _mm256_storeu_ps(&C[i * N + j + 56], c7);
                }

                for (; j <= N - 8; j += 8) {
                    __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                    c_vec = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j]), c_vec);
                    _mm256_storeu_ps(&C[i * N + j], c_vec);
                }

                for (; j < N; ++j) {
                    C[i * N + j] += a_val * B[k * N + j];
                }
            }
        }
    }
    std::string getName() const override { return "SIMD + Unroll 8x (Float)"; }
};

template <>
class Simd_Unroll_Solver<double> : public Matrix_Solver<double> {
public:
    void multiply(int M, int N, int K, const double* A, const double* B, double* C) override {
        
        // Pulizia C
        for (int i = 0; i < M * N; ++i) C[i] = 0.0;

        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                
                double a_val = A[i * K + k];
                __m256d a_vec = _mm256_set1_pd(a_val);

                int j = 0;

                  
                for (; j <= N - 32; j += 32) {
                    
                    __m256d c0 = _mm256_loadu_pd(&C[i * N + j + 0]);
                    __m256d c1 = _mm256_loadu_pd(&C[i * N + j + 4]);
                    __m256d c2 = _mm256_loadu_pd(&C[i * N + j + 8]);
                    __m256d c3 = _mm256_loadu_pd(&C[i * N + j + 12]);
                    __m256d c4 = _mm256_loadu_pd(&C[i * N + j + 16]);
                    __m256d c5 = _mm256_loadu_pd(&C[i * N + j + 20]);
                    __m256d c6 = _mm256_loadu_pd(&C[i * N + j + 24]);
                    __m256d c7 = _mm256_loadu_pd(&C[i * N + j + 28]);

                    c0 = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j + 0]), c0);
                    c1 = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j + 4]), c1);
                    c2 = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j + 8]), c2);
                    c3 = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j + 12]), c3);
                    c4 = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j + 16]), c4);
                    c5 = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j + 20]), c5);
                    c6 = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j + 24]), c6);
                    c7 = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j + 28]), c7);

                    _mm256_storeu_pd(&C[i * N + j + 0], c0);
                    _mm256_storeu_pd(&C[i * N + j + 4], c1);
                    _mm256_storeu_pd(&C[i * N + j + 8], c2);
                    _mm256_storeu_pd(&C[i * N + j + 12], c3);
                    _mm256_storeu_pd(&C[i * N + j + 16], c4);
                    _mm256_storeu_pd(&C[i * N + j + 20], c5);
                    _mm256_storeu_pd(&C[i * N + j + 24], c6);
                    _mm256_storeu_pd(&C[i * N + j + 28], c7);
                }

                for (; j <= N - 4; j += 4) {
                    __m256d c_vec = _mm256_loadu_pd(&C[i * N + j]);
                    c_vec = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j]), c_vec);
                    _mm256_storeu_pd(&C[i * N + j], c_vec);
                }

                for (; j < N; ++j) {
                    C[i * N + j] += a_val * B[k * N + j];
                }
            }
        }
    }
    std::string getName() const override { return "SIMD + Unroll 8x (Double)"; }
};


#endif
