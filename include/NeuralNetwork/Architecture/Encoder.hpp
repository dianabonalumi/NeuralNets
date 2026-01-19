#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "Architecture.hpp"
#include "../Layer/Layer.hpp"
#include "../Layer/LSTM.hpp"
#include "../Layer/Dense.hpp"
#include "../Matrix.hpp"
#include "../../matrix_solver.hpp"
#include "../Optimizer/Optimizer.hpp"

#include <vector>
#include <memory>
#include <string>

template <typename T>
class Encoder : public Architecture<T> { 
private:
    const std::shared_ptr<Matrix_Solver<T>> solver_;
    const std::shared_ptr<Optimizer<T>> optimizer_;
    const int window_, stride_;
    const int in_shape_, out_shape_, hidden_state_;
    
    LSTM<T> lstm_layer_;
    Dense<T> dense_layer_;

    Matrix<T> grad;

public:
    Encoder(const std::shared_ptr<Matrix_Solver<T>>& solver,
        const std::shared_ptr<Optimizer<T>>& optimizer,
        const int window,
        const int stride,
        const int in_shape,
        const int out_shape,
        const int hidden_state) 
        : solver_(solver), optimizer_(optimizer), in_shape_(in_shape), out_shape_(out_shape),
        hidden_state_(hidden_state), window_(window), stride_(stride),
        lstm_layer_(solver, optimizer, in_shape, hidden_state),
        dense_layer_(solver, optimizer, hidden_state, out_shape) {
            lstm_layer_.WeightInitialization(WeightInit::He);
            dense_layer_.WeightInitialization(WeightInit::He);
        }
    
    //aggiungo metodo per estrarre finestra
    Matrix<T> getWindow(const Matrix<T>& full_signal, int start_idx) {
        Matrix<T> window_mat(1, window_); 
        
        const T* src = full_signal.Flatten();
        T* dst = window_mat.Flatten();
        
        for(int i=0; i<window_; ++i) {
            if(start_idx + i < in_shape_) {   
                dst[i] = src[start_idx + i];
            } else {
                dst[i] = 0; 
            }
        }
        return window_mat; 
    }


    Matrix<T> Predict(const Matrix<T>& X) {
        // calcolo il numero di finestre
        int num_windows = (in_shape_ - window_) / stride_ + 1;
        if (num_windows <= 0) num_windows = 1;
        
        
        Matrix<T> latent_output(1, out_shape_);
        
        int window_idx = 0;
        Matrix<T> lstm_out;
        this->lstm_layer_.resetState();
        for (int i = 0; i <= in_shape_ - window_; i += stride_) {
            Matrix<T> x_window = getWindow(X, i);

            lstm_out = lstm_layer_.Forward(x_window);
        }

        latent_output = dense_layer_.Forward(lstm_out);

        return latent_output;
    }

    Matrix<T> Eval(const Matrix<T>& target) { return target; }

    void SetGradient(const Matrix<T>& grad) {
        this->grad = grad;
    }

    Matrix<T> Backward() {
        Matrix<T> g = dense_layer_.Backward(grad);

        g = lstm_layer_.Backward(g);

        return g;
    }
};

#endif
