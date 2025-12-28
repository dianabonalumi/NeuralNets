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


}