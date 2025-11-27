#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "Layer.hpp"

template <typename T>
class Sigmoid : public Layer<T> {
public:
    Matrix<T> Forward(const Matrix<T>);
    Matrix<T> Backward(const Matrix<T>);
};

#endif