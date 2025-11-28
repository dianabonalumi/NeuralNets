#ifndef RELU_HPP
#define RELU_HPP

#include "Layer.hpp"

template <typename T>
class ReLU : public Layer<T> {
public:
    // Constructor
    ReLU(std::shared_ptr<Matrix_Solver<T>> solver) : Layer<T>(solver) {}

    // Forward Pass (Placeholder: returns input as-is)
    Matrix<T> Forward(const Matrix<T> X) override {
        return X; 
    }

    // Backward Pass (Placeholder: returns gradient as-is)
    Matrix<T> Backward(const Matrix<T> grad) override {
        return grad;
    }
};

#endif