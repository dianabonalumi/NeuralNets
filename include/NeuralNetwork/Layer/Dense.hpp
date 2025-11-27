#ifndef DENSE_HPP
#define DENSE_HPP

#include "Layer.hpp"

template <typename T>
class Dense : public Layer<T> {
private:
    int in, out;
    Matrix<T> weights;
public:
    Dense(std::shared_ptr<Matrix_Solver<T>> solver, int in, int out)
        :Layer<T>(solver), in(in), out(out) {}
    
    Dense(std::shared_ptr<Matrix_Solver<T>> solver, int in, int out, const Matrix<T> weights) 
        :Dense<T>(solver, in, out) {
            this->weights = weights;
        }

    Matrix<T> Forward(const Matrix<T> X) {
        // Dense forward pass
        return X;
    }
    Matrix<T> Backward(const Matrix<T> grad) {
        // Dense backward pass
        return grad;
    }
};

#endif