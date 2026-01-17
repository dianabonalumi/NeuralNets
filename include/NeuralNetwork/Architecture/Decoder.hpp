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
    const int in_shape_, out_length_;

    Dense<T> dense_layer_;

    Matrix<T> grad;

public:
    Decoder(const std::shared_ptr<Matrix_Solver<T>>& solver,
        const std::shared_ptr<Optimizer<T>>& optimizer,
        const int in_shape,
        const int out_length) 
        : solver_(solver), optimizer_(optimizer), in_shape_(in_shape), out_length_(out_length),
        dense_layer_(solver, optimizer, in_shape, out_length) {
            dense_layer_.WeightInitialization(WeightInit::He);
        }

    Matrix<T> Predict(const Matrix<T>& X) {
        Matrix<T> result;

        result = this->dense_layer_.Forward(X);

        return result;
    }

    Matrix<T> Eval(const Matrix<T>& target) { return target; }

    void SetGradient(const Matrix<T>& grad) {
        this->grad = grad;
    }

    Matrix<T> Backward() {
        Matrix<T> g = dense_layer_.Backward(grad);

        return g;
    }

    T* Serialize() {
        T* dense_data = dense_layer_.Serialize();
        
        return dense_data;
    }

    void Deserialize(T* data) {
        if (!data) return;
        
        dense_layer_.Deserialize(data);
    }
};

#endif
