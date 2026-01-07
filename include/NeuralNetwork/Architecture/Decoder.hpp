#ifndef DECODER_HPP
#define DECODER_HPP

#include "Architecture.hpp"
#include "../Layer/Layer.hpp"
#include "../Layer/LSTM.hpp"
#include "../Layer/Dense.hpp"
#include "../Matrix.hpp"
#include "../../matrix_solver.hpp"
#include "../Optimizer/Optimizer.hpp"

#include <vector>
#include <memory>

template <typename T>
class Decoder : public Architecture<T> { 
private:
    const std::shared_ptr<Matrix_Solver<T>> solver_;
    const std::shared_ptr<Optimizer<T>> optimizer_;
    const int in_shape_, out_length_, hidden_state_;
    
    LSTM<T> lstm_layer_;
    Dense<T> dense_layer_;

    Matrix<T> grad;

public:
    Decoder(const std::shared_ptr<Matrix_Solver<T>>& solver,
        const std::shared_ptr<Optimizer<T>>& optimizer,
        const int in_shape,
        const int out_length,
        const int hidden_state) 
        : solver_(solver), optimizer_(optimizer), in_shape_(in_shape), out_length_(out_length),
        hidden_state_(hidden_state),
        lstm_layer_(solver, optimizer, in_shape, hidden_state),
        dense_layer_(solver, optimizer, hidden_state, out_length) {
            lstm_layer_.WeightInitialization(WeightInit::He);
            dense_layer_.WeightInitialization(WeightInit::He);
        }

    Matrix<T> Predict(const Matrix<T>& X) {
        Matrix<T> result(1, this->out_length_);

        this->lstm_layer_.resetState();
        Matrix<T> c = this->lstm_layer_.Forward(X);
        for(int i = 0;i < this->out_length_;i++) {
            c = this->lstm_layer_.Forward(c);
        }
        result = this->dense_layer_.Forward(c);

        return result;
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

    T* Serialize() {
        // Serialize LSTM then Dense
        T* lstm_data = lstm_layer_.Serialize();
        T* dense_data = dense_layer_.Serialize();
        
        size_t lstm_size = static_cast<size_t>(lstm_data[0]) + 1;
        size_t dense_size = static_cast<size_t>(dense_data[0]) + 1;
        size_t total_size = lstm_size + dense_size;
        
        T* buffer = new T[total_size + 1];
        buffer[0] = static_cast<T>(total_size);
        
        // Copy LSTM data
        for(size_t i = 0; i < lstm_size; ++i) {
            buffer[i + 1] = lstm_data[i];
        }
        
        // Copy Dense data
        for(size_t i = 0; i < dense_size; ++i) {
            buffer[lstm_size + i + 1] = dense_data[i];
        }
        
        delete[] lstm_data;
        delete[] dense_data;
        
        return buffer;
    }

    void Deserialize(T* data) {
        if (!data) return;
        
        size_t lstm_size = static_cast<size_t>(data[1]) + 1;
        
        // Deserialize LSTM
        T* lstm_buffer = new T[lstm_size];
        for(size_t i = 0; i < lstm_size; ++i) {
            lstm_buffer[i] = data[i + 1];
        }
        lstm_layer_.Deserialize(lstm_buffer);
        delete[] lstm_buffer;
        
        // Deserialize Dense
        T* dense_buffer = &data[lstm_size + 1];
        dense_layer_.Deserialize(dense_buffer);
    }
};

#endif
