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

    // Reset dello stato che viene chiamato all'inizio di una nuova epoca
    void resetState() {
        // Mettiamo le dimensioni a 0 per forzare la ri-inizializzazione nel Forward
        h_state = Matrix<T>(0, 0);
        c_state = Matrix<T>(0, 0);
    }

    // Funzioni di attivazione helper (inline per brevità)
    T sigmoid(T x) { return 1.0 / (1.0 + std::exp(-x)); }
    T tanh_act(T x) { return std::tanh(x); }


}