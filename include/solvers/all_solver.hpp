#ifndef ALL_SOLVER_HPP
#define ALL_SOLVER_HPP

#include <omp.h>
#include <immintrin.h>
#include "../matrix_solver.hpp"

#define TILE_M 64
#define TILE_N_FLOAT 128
#define TILE_N_DOUBLE 64
#define TILE_K 64

template <typename T>
class All_Solver : public Matrix_Solver<T> {
public:
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {}
    std::string getName() const override { return "SIMD + Unroll + Tiling + OpenMP Solver"; }
};

template <>
class All_Solver<float> : public Matrix_Solver<float> {
public:
    void multiply(int M, int N, int K, const float* A, const float* B, float* C) override {

        // OpenMP Parallelization
        #pragma omp parallel for collapse(2)
        for (int ii = 0; ii < M; ii += TILE_M) {
            for (int jj = 0; jj < N; jj += TILE_N_FLOAT) {

                // Calculate boundaries for current tile
                int i_end = std::min(ii + TILE_M, M);
                int j_end = std::min(jj + TILE_N_FLOAT, N);

                // Initialize specific block of C to 0.0f
                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        C[i * N + j] = 0.0f;
                    }
                }

                // K-Dimension Tiling
                for (int kk = 0; kk < K; kk += TILE_K) {
                    int k_end = std::min(kk + TILE_K, K);
                    int i = ii;

                    // Process 2 rows of A at a time
                    for (; i <= i_end - 2; i += 2) {

                        for (int k = kk; k < k_end; ++k) {

                            __m256 a_vec_row0 = _mm256_set1_ps(A[i * K + k]);
                            __m256 a_vec_row1 = _mm256_set1_ps(A[(i + 1) * K + k]);

                            int j = jj; // Start j at the tile start (jj)

                            // Unroll factor 32
                            for (; j <= j_end - 32; j += 32) {
                                // Load B
                                __m256 b0 = _mm256_loadu_ps(&B[k * N + j + 0]);
                                __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);
                                __m256 b2 = _mm256_loadu_ps(&B[k * N + j + 16]);
                                __m256 b3 = _mm256_loadu_ps(&B[k * N + j + 24]);

                                // Load C (Row i)
                                __m256 c0_r0 = _mm256_loadu_ps(&C[i * N + j + 0]);
                                __m256 c1_r0 = _mm256_loadu_ps(&C[i * N + j + 8]);
                                __m256 c2_r0 = _mm256_loadu_ps(&C[i * N + j + 16]);
                                __m256 c3_r0 = _mm256_loadu_ps(&C[i * N + j + 24]);

                                // FMADD (Row i)
                                c0_r0 = _mm256_fmadd_ps(a_vec_row0, b0, c0_r0);
                                c1_r0 = _mm256_fmadd_ps(a_vec_row0, b1, c1_r0);
                                c2_r0 = _mm256_fmadd_ps(a_vec_row0, b2, c2_r0);
                                c3_r0 = _mm256_fmadd_ps(a_vec_row0, b3, c3_r0);

                                // Store C (Row i)
                                _mm256_storeu_ps(&C[i * N + j + 0], c0_r0);
                                _mm256_storeu_ps(&C[i * N + j + 8], c1_r0);
                                _mm256_storeu_ps(&C[i * N + j + 16], c2_r0);
                                _mm256_storeu_ps(&C[i * N + j + 24], c3_r0);

                                // Load C (Row i+1)
                                __m256 c0_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j + 0]);
                                __m256 c1_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j + 8]);
                                __m256 c2_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j + 16]);
                                __m256 c3_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j + 24]);

                                // FMADD (Row i+1)
                                c0_r1 = _mm256_fmadd_ps(a_vec_row1, b0, c0_r1);
                                c1_r1 = _mm256_fmadd_ps(a_vec_row1, b1, c1_r1);
                                c2_r1 = _mm256_fmadd_ps(a_vec_row1, b2, c2_r1);
                                c3_r1 = _mm256_fmadd_ps(a_vec_row1, b3, c3_r1);

                                // Store C (Row i+1)
                                _mm256_storeu_ps(&C[(i + 1) * N + j + 0], c0_r1);
                                _mm256_storeu_ps(&C[(i + 1) * N + j + 8], c1_r1);
                                _mm256_storeu_ps(&C[(i + 1) * N + j + 16], c2_r1);
                                _mm256_storeu_ps(&C[(i + 1) * N + j + 24], c3_r1);
                            }

                            // Unroll factor 8
                            for (; j <= j_end - 8; j += 8) {
                                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);

                                __m256 c_r0 = _mm256_loadu_ps(&C[i * N + j]);
                                c_r0 = _mm256_fmadd_ps(a_vec_row0, b_vec, c_r0);
                                _mm256_storeu_ps(&C[i * N + j], c_r0);

                                __m256 c_r1 = _mm256_loadu_ps(&C[(i + 1) * N + j]);
                                c_r1 = _mm256_fmadd_ps(a_vec_row1, b_vec, c_r1);
                                _mm256_storeu_ps(&C[(i + 1) * N + j], c_r1);
                            }

                            // Scalar cleanup for J
                            for (; j < j_end; ++j) {
                                C[i * N + j] += A[i * K + k] * B[k * N + j];
                                C[(i + 1) * N + j] += A[(i + 1) * K + k] * B[k * N + j];
                            }
                        }
                    }

                    // Handle remaining odd row of A (if any within this block)
                    for (; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            float a_val = A[i * K + k];
                            __m256 a_vec = _mm256_set1_ps(a_val);

                            int j = jj;

                            for (; j <= j_end - 8; j += 8) {
                                __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                                c_vec = _mm256_fmadd_ps(a_vec, _mm256_loadu_ps(&B[k * N + j]), c_vec);
                                _mm256_storeu_ps(&C[i * N + j], c_vec);
                            }
                            for (; j < j_end; ++j) {
                                C[i * N + j] += a_val * B[k * N + j];
                            }
                        }
                    }
                } // End of KK loop
            } // End of JJ loop
        } // End of II loop
    }

    std::string getName() const override {
        return "SIMD + Unroll 2D + Tiling + OpenMP (Float)";
    }
};


template <>
class All_Solver<double> : public Matrix_Solver<double> {
public:
    void multiply(int M, int N, int K, const double* A, const double* B, double* C) override {

        // OpenMP Parallelization
#       pragma omp parallel for collapse(2)
        for (int ii = 0; ii < M; ii += TILE_M) {
            for (int jj = 0; jj < N; jj += TILE_N_DOUBLE) {

                // Determine boundaries
                int i_end = std::min(ii + TILE_M, M);
                int j_end = std::min(jj + TILE_N_DOUBLE, N);

                // Initialize C block to 0.0ù
                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        C[i * N + j] = 0.0;
                    }
                }

                // Loop over K tiles
                for (int kk = 0; kk < K; kk += TILE_K) {
                    int k_end = std::min(kk + TILE_K, K);

                    int i = ii;
                    for (; i <= i_end - 2; i += 2) {

                        for (int k = kk; k < k_end; ++k) {

                            // Broadcast A values
                            __m256d a_vec_row0 = _mm256_set1_pd(A[i * K + k]);
                            __m256d a_vec_row1 = _mm256_set1_pd(A[(i + 1) * K + k]);

                            int j = jj;

                            // Unroll 16 (4 vectors of doubles)
                            for (; j <= j_end - 16; j += 16) {

                                // Load B
                                __m256d b0 = _mm256_loadu_pd(&B[k * N + j + 0]);
                                __m256d b1 = _mm256_loadu_pd(&B[k * N + j + 4]);
                                __m256d b2 = _mm256_loadu_pd(&B[k * N + j + 8]);
                                __m256d b3 = _mm256_loadu_pd(&B[k * N + j + 12]);

                                // Load C (Row i)
                                __m256d c0_r0 = _mm256_loadu_pd(&C[i * N + j + 0]);
                                __m256d c1_r0 = _mm256_loadu_pd(&C[i * N + j + 4]);
                                __m256d c2_r0 = _mm256_loadu_pd(&C[i * N + j + 8]);
                                __m256d c3_r0 = _mm256_loadu_pd(&C[i * N + j + 12]);

                                // FMA (Row i)
                                c0_r0 = _mm256_fmadd_pd(a_vec_row0, b0, c0_r0);
                                c1_r0 = _mm256_fmadd_pd(a_vec_row0, b1, c1_r0);
                                c2_r0 = _mm256_fmadd_pd(a_vec_row0, b2, c2_r0);
                                c3_r0 = _mm256_fmadd_pd(a_vec_row0, b3, c3_r0);

                                // Store C (Row i)
                                _mm256_storeu_pd(&C[i * N + j + 0], c0_r0);
                                _mm256_storeu_pd(&C[i * N + j + 4], c1_r0);
                                _mm256_storeu_pd(&C[i * N + j + 8], c2_r0);
                                _mm256_storeu_pd(&C[i * N + j + 12], c3_r0);

                                // Load C (Row i+1)
                                __m256d c0_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j + 0]);
                                __m256d c1_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j + 4]);
                                __m256d c2_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j + 8]);
                                __m256d c3_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j + 12]);

                                // FMA (Row i+1)
                                c0_r1 = _mm256_fmadd_pd(a_vec_row1, b0, c0_r1);
                                c1_r1 = _mm256_fmadd_pd(a_vec_row1, b1, c1_r1);
                                c2_r1 = _mm256_fmadd_pd(a_vec_row1, b2, c2_r1);
                                c3_r1 = _mm256_fmadd_pd(a_vec_row1, b3, c3_r1);

                                // Store C (Row i+1)
                                _mm256_storeu_pd(&C[(i + 1) * N + j + 0], c0_r1);
                                _mm256_storeu_pd(&C[(i + 1) * N + j + 4], c1_r1);
                                _mm256_storeu_pd(&C[(i + 1) * N + j + 8], c2_r1);
                                _mm256_storeu_pd(&C[(i + 1) * N + j + 12], c3_r1);
                            }

                            // Unroll 4 (1 vector of doubles)
                            for (; j <= j_end - 4; j += 4) {
                                __m256d b_vec = _mm256_loadu_pd(&B[k * N + j]);

                                __m256d c_r0 = _mm256_loadu_pd(&C[i * N + j]);
                                c_r0 = _mm256_fmadd_pd(a_vec_row0, b_vec, c_r0);
                                _mm256_storeu_pd(&C[i * N + j], c_r0);

                                __m256d c_r1 = _mm256_loadu_pd(&C[(i + 1) * N + j]);
                                c_r1 = _mm256_fmadd_pd(a_vec_row1, b_vec, c_r1);
                                _mm256_storeu_pd(&C[(i + 1) * N + j], c_r1);
                            }

                            // Scalar Cleanup
                            for (; j < j_end; ++j) {
                                C[i * N + j] += A[i * K + k] * B[k * N + j];
                                C[(i + 1) * N + j] += A[(i + 1) * K + k] * B[k * N + j];
                            }
                        }
                    }
                    for (; i < i_end; ++i) {
                        for (int k = kk; k < k_end; ++k) {
                            double a_val = A[i * K + k];
                            __m256d a_vec = _mm256_set1_pd(a_val);

                            int j = jj;

                            // Vector loop
                            for (; j <= j_end - 4; j += 4) {
                                __m256d c_vec = _mm256_loadu_pd(&C[i * N + j]);
                                c_vec = _mm256_fmadd_pd(a_vec, _mm256_loadu_pd(&B[k * N + j]), c_vec);
                                _mm256_storeu_pd(&C[i * N + j], c_vec);
                            }

                            // Scalar cleanup
                            for (; j < j_end; ++j) {
                                C[i * N + j] += a_val * B[k * N + j];
                            }
                        }
                    }
                } // End KK
            } // End JJ
        } // End II
    }
    std::string getName() const override { return "SIMD + Unroll 2D + Tiling + OpenMP (Double)"; }
};

#endif
