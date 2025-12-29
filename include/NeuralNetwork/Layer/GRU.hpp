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

    // Cache per Backpropagation
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
    void resetState() {
        h_state = Matrix<T>(0, 0);
    }

    // Implementazione obbligatoria per classe astratta Layer
    void WeightInitialization(const WeightInit& technique) override {
        // Qui andrebbe l'inizializzazione random (Xavier/He)
        // Per ora lasciamo vuoto o implementabile in futuro
    }

    // Helpers
    inline T sigmoid(T x) { return 1.0 / (1.0 + std::exp(-x)); }
    inline T tanh_act(T x) { return std::tanh(x); }
    inline T d_sigmoid(T y) { return y * (1.0 - y); }
    inline T d_tanh(T y) { return 1.0 - (y * y); }

    // --- FORWARD PASS ---
    Matrix<T> Forward(const Matrix<T>& X) override {
        size_t batch_size = X.rows();
        input_cache = X; // Salva input per backward

        // Init stato se dimensione cambia
        if (h_state.rows() != batch_size || h_state.cols() != hidden_size) {
            h_state = Matrix<T>(batch_size, hidden_size);
            
        }
        prev_h_state = h_state;

        // Output matrix
        Matrix<T> next_h(batch_size, hidden_size);

        // Allocazione cache
        cache_z = Matrix<T>(batch_size, hidden_size);
        cache_r = Matrix<T>(batch_size, hidden_size);
        cache_n = Matrix<T>(batch_size, hidden_size);
        cache_Uh_prev = Matrix<T>(batch_size, hidden_size);

        
        // Calcoliamo X*W e H*U per tutte le porte
        
        // Update Gate (z)
        Matrix<T> XW_z(batch_size, hidden_size), hU_z(batch_size, hidden_size);
        this->solver_->multiply(batch_size, hidden_size, input_features, X.Flatten(), W_z.Flatten(), XW_z.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, prev_h_state.Flatten(), U_z.Flatten(), hU_z.Flatten());

        // Reset Gate (r)
        Matrix<T> XW_r(batch_size, hidden_size), hU_r(batch_size, hidden_size);
        this->solver_->multiply(batch_size, hidden_size, input_features, X.Flatten(), W_r.Flatten(), XW_r.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, prev_h_state.Flatten(), U_r.Flatten(), hU_r.Flatten());

        // Candidate (n) - Parte Input (X*W)
        Matrix<T> XW_n(batch_size, hidden_size);
        this->solver_->multiply(batch_size, hidden_size, input_features, X.Flatten(), W_n.Flatten(), XW_n.Flatten());
        
        // Candidate (n) - Parte Ricorrente (H*U)
    
        Matrix<T> hU_n(batch_size, hidden_size);
        this->solver_->multiply(batch_size, hidden_size, hidden_size, prev_h_state.Flatten(), U_n.Flatten(), hU_n.Flatten());
        
        // Salviamo hU_n grezzo nella cache perché serve per il gradiente del reset gate
        cache_Uh_prev = hU_n; 

        