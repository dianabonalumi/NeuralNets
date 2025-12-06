#ifndef SIMD_SOLVER_HPP
#define SIMD_SOLVER_HPP

#include <immintrin.h> // Header fondamentale per AVX
#include "../matrix_solver.hpp"


 // require flag -mfma for FMA instructions
//generic declaration, specializations follow
template <typename T>
class Simd_Solver : public Matrix_Solver<T> {
public:
    void multiply(int M, int N, int K, const T* A, const T* B, T* C) override {
        // Fallback or error if T not float/double
    }
    std::string getName() const override { return "SIMD Solver"; }
};

// it follow one specialization for float and double
template <>
class Simd_Solver<float> : public Matrix_Solver<float> {
public:
    void multiply(int M, int N, int K, const float* A, const float* B, float* C) override {
        
        // initialization of C at zero
        for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

        // change order of the loops for better performance (coalesced memory access)
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                
               //SIMD load A[i][k] into all elements of a_vec
                float a_val = A[i * K + k];
                __m256 a_vec = _mm256_set1_ps(a_val);

                int j = 0;
                for (; j <= N - 8; j += 8) {

                    __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                    
                    __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                    
                    // FMA: c_vec = c_vec + (a_vec * b_vec)
                    // _mm256_fmadd_ps is faster of mul + add separately
                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                    _mm256_storeu_ps(&C[i * N + j], c_vec);
                }

                //cleanup for remaining elements
                for (; j < N; ++j) {
                    C[i * N + j] += a_val * B[k * N + j];
                }
            }
        }
    }
    std::string getName() const override { return "SIMD Solver (Float)"; }
};

//same as above but for double so the step is 4 instead of 8
template <>
class Simd_Solver<double> : public Matrix_Solver<double> {
public:
    void multiply(int M, int N, int K, const double* A, const double* B, double* C) override {
        
        for (int i = 0; i < M * N; ++i) C[i] = 0.0;

        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                
                double a_val = A[i * K + k];
                __m256d a_vec = _mm256_set1_pd(a_val); 

                int j = 0;
                for (; j <= N - 4; j += 4) {
                    __m256d c_vec = _mm256_loadu_pd(&C[i * N + j]);
                    __m256d b_vec = _mm256_loadu_pd(&B[k * N + j]);
                    
                    c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);

                    _mm256_storeu_pd(&C[i * N + j], c_vec);
                }

                for (; j < N; ++j) {
                    C[i * N + j] += a_val * B[k * N + j];
                }
            }
        }
    }
    std::string getName() const override { return "SIMD Solver (Double)"; }
};

#endif
