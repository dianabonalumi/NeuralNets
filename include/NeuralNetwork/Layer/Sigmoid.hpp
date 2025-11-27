#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "Layer.hpp"

template <typename T>
class Sigmoid : public Layer<T> {
public:
    Sigmoid(std::shared_ptr<Matrix_Solver<T>> solver): Layer<T>(solver) {}

    Matrix<T> Forward(const Matrix<T> X) {
        // Sigmoid forward pass
        return X;
    }

    Matrix<T> Backward(const Matrix<T> grad) {
        // Sigmoid backward pass
        return grad;
    }
};

#endif