#ifndef GRU_HPP
#define GRU_HPP

#include "Layer.hpp"
#include <cmath>
#include <vector>

template <typename T> 
class GRU : public Layer<T> {
private: 
    int input_features;
    int hidden_size;

    
    Matrix<T> h_state; 
    Matrix<T> prev_h_state;

    // Cache input per backward
    Matrix<T> input_cache;

    //Pesi e bias
    
    // Update Gate (z)
    Matrix<T> W_z, U_z, b_z;
    // Reset Gate (r)
    Matrix<T> W_r, U_r, b_r;
    // New Memory / Candidate (n) o (h_tilde)
    Matrix<T> W_n, U_n, b_n;

    // Cache per Backpropagation ---
    Matrix<T> cache_z; // Update gate output
    Matrix<T> cache_r; // Reset gate output
    Matrix<T> cache_n; // Candidate activation
    // Cache per il termine (U_n * h_prev) che serve nel calcolo del candidato
    Matrix<T> cache_Uh_prev;

    public: 
    GRU(const std::shared_ptr<Matrix_Solver<T>>& solver, 
        const std::shared_ptr<Optimizer<T>>& optimizer, 
        int input_features, 
        int hidden_size)
        : Layer<T>(solver, optimizer), input_features(input_features), hidden_size(hidden_size) 
    {
        initMatrices();
    }

    void initMatrices() {
        // Pesi Input (Input x Hidden)
        W_z = Matrix<T>(input_features, hidden_size);
        W_r = Matrix<T>(input_features, hidden_size);
        W_n = Matrix<T>(input_features, hidden_size);

        // Pesi Ricorrenti (Hidden x Hidden)
        U_z = Matrix<T>(hidden_size, hidden_size);
        U_r = Matrix<T>(hidden_size, hidden_size);
        U_n = Matrix<T>(hidden_size, hidden_size);

        // Bias (1 x Hidden)
        b_z = Matrix<T>(1, hidden_size);
        b_r = Matrix<T>(1, hidden_size);
        b_n = Matrix<T>(1, hidden_size);

        h_state = Matrix<T>(0, 0);
    }