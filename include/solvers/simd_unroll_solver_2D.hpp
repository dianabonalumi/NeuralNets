#ifndef SIMD_UNROLL_SOLVER_2D_HPP
#define SIMD_UNROLL_SOLVER_2D_HPP

#include <immintrin.h>
#include "../matrix_solver.hpp"

template <typename T>
class Simd_Unroll_Solver_2D : public Matrix_Solver<T> {
public:
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {}
    std::string getName() const override { return "SIMD + Unroll Solver"; }
};

template <>
class Simd_Unroll_Solver_2D<float> : public Matrix_Solver<float> {
public:
    void multiply(int M, int N, int K, const float* A, const float* B, float* C) override {
        
        for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

        int i = 0;
        for (; i <= M - 2; i += 2) {
            
            for (int k = 0; k < K; ++k) {
                
                __m256 a_vec_row0 = _mm256_set1_ps(A[i * K + k]);       // A[i][k]
                __m256 a_vec_row1 = _mm256_set1_ps(A[(i + 1) * K + k]); // A[i+1][k]

                int j = 0;
                
                       
                for (; j <= N - 32; j += 32) {
                    
                    __m256 b0 = _mm256_loadu_ps(&B[k * N + j + 0]);
                    __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);
                    __m256 b2 = _mm256_loadu_ps(&B[k * N + j + 16]);
                    __m256 b3 = _mm256_loadu_ps(&B[k * N + j + 24]);

                    __m256 c0_r0 = _mm256_loadu_ps(&C[i * N + j + 0]);
                    __m256 c1_r0 = _mm256_loadu_ps(&C[i * N + j + 8]);
                    __m256 c2_r0 = _mm256_loadu_ps(&C[i * N + j + 16]);
                    __m256 c3_r0 = _mm256_loadu_ps(&C[i * N + j + 24]);

                    c0_r0 = _mm256_fmadd_ps(a_vec_row0, b0, c0_r0);
                    c1_r0 = _mm256_fmadd_ps(a_vec_row0, b1, c1_r0);
                    c2_r0 = _mm256_fmadd_ps(a_vec_row0, b2, c2_r0);
                    c3_r0 = _mm256_fmadd_ps(a_vec_row0, b3, c3_r0);

                    _mm256_storeu_ps(&C[i * N + j + 0], c0_r0);
                    _mm256_storeu_ps(&C[i * N + j + 8], c1_r0);
                    _mm256_storeu_ps(&C[i * N + j + 16], c2_r0);
                    _mm256_storeu_ps(&C[i * N + j + 24], c3_r0);

                    __m256 c0_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j + 0]);
                    __m256 c1_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j + 8]);
                    __m256 c2_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j + 16]);
                    __m256 c3_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j + 24]);

                    c0_r1 = _mm256_fmadd_ps(a_vec_row1, b0, c0_r1);
                    c1_r1 = _mm256_fmadd_ps(a_vec_row1, b1, c1_r1);
                    c2_r1 = _mm256_fmadd_ps(a_vec_row1, b2, c2_r1);
                    c3_r1 = _mm256_fmadd_ps(a_vec_row1, b3, c3_r1);

                    _mm256_storeu_ps(&C[(i + 1) * N + j + 0], c0_r1);
                    _mm256_storeu_ps(&C[(i + 1) * N + j + 8], c1_r1);
                    _mm256_storeu_ps(&C[(i + 1) * N + j + 16], c2_r1);
                    _mm256_storeu_ps(&C[(i + 1) * N + j + 24], c3_r1);
                }
                
                for (; j <= N - 8; j += 8) {
                    __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                    
                    __m256 c_r0 = _mm256_loadu_ps(&C[i * N + j]);
                    c_r0 = _mm256_fmadd_ps(a_vec_row0, b_vec, c_r0);
                    _mm256_storeu_ps(&C[i * N + j], c_r0);

                    __m256 c_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j]);
                    c_r1 = _mm256_fmadd_ps(a_vec_row1, b_vec, c_r1);
                    _mm256_storeu_ps(&C[(i + 1) * N + j], c_r1);
                }

                for (; j < N; ++j) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                    C[(i + 1) * N + j] += A[(i + 1) * K + k] * B[k * N + j];
                }
            }
        }

        for (; i < M; ++i) {
             for (int k = 0; k < K; ++k) {
                float a_val = A[i * K + k];
                __m256 a_vec = _mm256_set1_ps(a_val);
                int j = 0;
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

std::string getName() const override { 
    return "SIMD Unroll 2D (Float)"; 
}
};


template <>
class Simd_Unroll_Solver_2D<double> : public Matrix_Solver<double> {
public:
    void multiply(int M, int N, int K, const double* A, const double* B, double* C) override {
        
        for (int i = 0; i < M * N; ++i) C[i] = 0.0;

        int i = 0;
        for (; i <= M - 2; i += 2) {
            
            for (int k = 0; k < K; ++k) {
                
                __m256d a_vec_row0 = _mm256_set1_pd(A[i * K + k]);       // A[i][k]
                __m256d a_vec_row1 = _mm256_set1_pd(A[(i + 1) * K + k]); // A[i+1][k]

                int j = 0;
                
                for (; j <= N - 16; j += 16) {
                    
                     __m256d b0 = _mm256_loadu_pd(&B[k * N + j + 0]);
                    __m256d b1 = _mm256_loadu_pd(&B[k * N + j + 4]);
                    __m256d b2 = _mm256_loadu_pd(&B[k * N + j + 8]);
                    __m256d b3 = _mm256_loadu_pd(&B[k * N + j + 12]);

                    __m256d c0_r0 = _mm256_loadu_pd(&C[i * N + j + 0]);
                    __m256d c1_r0 = _mm256_loadu_pd(&C[i * N + j + 4]);
                    __m256d c2_r0 = _mm256_loadu_pd(&C[i * N + j + 8]);
                    __m256d c3_r0 = _mm256_loadu_pd(&C[i * N + j + 12]);

                    c0_r0 = _mm256_fmadd_pd(a_vec_row0, b0, c0_r0);
                    c1_r0 = _mm256_fmadd_pd(a_vec_row0, b1, c1_r0);
                    c2_r0 = _mm256_fmadd_pd(a_vec_row0, b2, c2_r0);
                    c3_r0 = _mm256_fmadd_pd(a_vec_row0, b3, c3_r0);

                    _mm256_storeu_pd(&C[i * N + j + 0], c0_r0);
                    _mm256_storeu_pd(&C[i * N + j + 4], c1_r0);
                    _mm256_storeu_pd(&C[i * N + j + 8], c2_r0);
                    _mm256_storeu_pd(&C[i * N + j + 12], c3_r0);

                    __m256d c0_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j + 0]);
                    __m256d c1_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j + 4]);
                    __m256d c2_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j + 8]);
                    __m256d c3_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j + 12]);

                    c0_r1 = _mm256_fmadd_pd(a_vec_row1, b0, c0_r1);
                    c1_r1 = _mm256_fmadd_pd(a_vec_row1, b1, c1_r1);
                    c2_r1 = _mm256_fmadd_pd(a_vec_row1, b2, c2_r1);
                    c3_r1 = _mm256_fmadd_pd(a_vec_row1, b3, c3_r1);

                    _mm256_storeu_pd(&C[(i + 1) * N + j + 0], c0_r1);
                    _mm256_storeu_pd(&C[(i + 1) * N + j + 4], c1_r1);
                    _mm256_storeu_pd(&C[(i + 1) * N + j + 8], c2_r1);
                    _mm256_storeu_pd(&C[(i + 1) * N + j + 12], c3_r1);
                }
                
                for (; j <= N - 4; j += 4) {
                    __m256d b_vec = _mm256_loadu_pd(&B[k * N + j]);
                    
                    __m256d c_r0 = _mm256_loadu_pd(&C[i * N + j]);
                    c_r0 = _mm256_fmadd_pd(a_vec_row0, b_vec, c_r0);
                    _mm256_storeu_pd(&C[i * N + j], c_r0);

                    __m256d c_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j]);
                    c_r1 = _mm256_fmadd_pd(a_vec_row1, b_vec, c_r1);
                    _mm256_storeu_pd(&C[(i + 1) * N + j], c_r1);
                }

                for (; j < N; ++j) {
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
                    C[(i + 1) * N + j] += A[(i + 1) * K + k] * B[k * N + j];
                }
            }
        }

              for (; i < M; ++i) {
             for (int k = 0; k < K; ++k) {
                double a_val = A[i * K + k];
                __m256d a_vec = _mm256_set1_pd(a_val);
                int j = 0;
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
    std::string getName() const override { return "SIMD+Unroll 2D (Double)"; }
};

#endif 
