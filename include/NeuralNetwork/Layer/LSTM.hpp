#ifndef LSTM_HPP
#define LSTM_HPP

#include "Layer.hpp"
#include <cmath>
#include <vector>
#include <random>

template <typename T> 
class LSTM : public Layer<T> >{
private: 
    int input_features;
    int hidden_size;

    //definisco gli stati interni

    Matrix<T> h_state; // hidden state, memoria a breve termine
    Matrix<T> c_state; // Cell state, memoria a lungo termine

    // Pesi e bias: W sono i pesi di input, U i pesi ricorrenti, b bias
    
    //Forget Gate
    Matrix<T> W_f, U_f, b_f;

    //Input Gate
    Matrix<T> W_i, U_i, b_i;

    //Cell gate
    Matrix<T> W_c, U_c, b_c;

    //Output gate
    Matrix<T> W_o, U_o, b_o;

    //Cache per backpropagation
    Matrix<T> cache_f, cache_i, cache_o, cache_c_bar, cache_tanh_c;
    Matrix<T> prev_h_state, prev_c_state;

    public: 
    LSTM(const std::shared_ptr<Matrix_Solver<T>>& solver, 
         const std::shared_ptr<Optimizer<T>>& optimizer, 
         int input_features, 
         int hidden_size)
        : Layer<T>(solver, optimizer), input_features(input_features), hidden_size(hidden_size) 
    {
        // Inizializza pesi e stati
        initMatrices();
    }

    // Inizializzazione le dimensioni delle matrici, allocazione delo spazio per tutte le matrici create prima
    void initMatrices() {
        // Pesi Input (input_features x hidden_size)
        W_f = Matrix<T>(input_features, hidden_size);
        W_i = Matrix<T>(input_features, hidden_size);
        W_c = Matrix<T>(input_features, hidden_size);
        W_o = Matrix<T>(input_features, hidden_size);

        // Pesi Ricorrenti (hidden_size x hidden_size)
        U_f = Matrix<T>(hidden_size, hidden_size);
        U_i = Matrix<T>(hidden_size, hidden_size);
        U_c = Matrix<T>(hidden_size, hidden_size);
        U_o = Matrix<T>(hidden_size, hidden_size);

        // Bias (1 x hidden_size) - Broadcasted durante il calcolo
        b_f = Matrix<T>(1, hidden_size);
        b_i = Matrix<T>(1, hidden_size);
        b_c = Matrix<T>(1, hidden_size);
        b_o = Matrix<T>(1, hidden_size);

        // Gli stati iniziali sono a 0.
        h_state = Matrix<T>(0, 0);
        c_state = Matrix<T>(0, 0);
    }

    // Ogni volta che devo iniziare una nuova epoca faccio il reset dello stato
    void resetState() {
        // Mettiamo le dimensioni a 0 per forzare la ri-inizializzazione nel Forward
        h_state = Matrix<T>(0, 0);
        c_state = Matrix<T>(0, 0);
    }

    // Funzioni di attivazione helper (inline per brevità)
    T sigmoid(T x) { return 1.0 / (1.0 + std::exp(-x)); }
    T tanh_act(T x) { return std::tanh(x); }

    //FORWARD PASS
    Matrix<T> Forward(const Matrix<T>& X) override {
        size_t batch_size = X.rows();

        //Inizializzazione stati se è il primo step o se è cambiato il batch size
        if (h_state.rows() != batch_size || h_state.cols() != hidden_size) {
            h_state = Matrix<T>(batch_size, hidden_size); 
            c_state = Matrix<T>(batch_size, hidden_size);
        }

        // Salviamo gli stati precedenti per il backward
        prev_h_state = h_state; 
        prev_c_state = c_state;

        // Matrice risultato per il nuovo hidden state
        Matrix<T> next_h(batch_size, hidden_size);
        Matrix<T> next_c(batch_size, hidden_size);

        // matrici cache per il backward
        cache_f = Matrix<T>(batch_size, hidden_size);
        cache_i = Matrix<T>(batch_size, hidden_size);
        cache_c_bar = Matrix<T>(batch_size, hidden_size);
        cache_o = Matrix<T>(batch_size, hidden_size);
        cache_tanh_c = Matrix<T>(batch_size, hidden_size);

        //Calcolo di X*W e h*U per tutte e 4 le porte.

        // Matrici temporanee per i risultati intermedi
        Matrix<T> XW_f(batch_size, hidden_size), hU_f(batch_size, hidden_size);
        Matrix<T> XW_i(batch_size, hidden_size), hU_i(batch_size, hidden_size);
        Matrix<T> XW_c(batch_size, hidden_size), hU_c(batch_size, hidden_size);
        Matrix<T> XW_o(batch_size, hidden_size), hU_o(batch_size, hidden_size);

        // Flattening per passare i puntatori al solver
        const T* x_ptr = X.Flatten();
        const T* h_ptr = prev_h_state.Flatten();

        // Forget Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_f.Flatten(), XW_f.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_f.Flatten(), hU_f.Flatten());

        // Input Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_i.Flatten(), XW_i.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_i.Flatten(), hU_i.Flatten());

        //  Candidate (Cell) Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_c.Flatten(), XW_c.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_c.Flatten(), hU_c.Flatten());

        // Output Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_o.Flatten(), XW_o.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_o.Flatten(), hU_o.Flatten());


    }
}

       
        