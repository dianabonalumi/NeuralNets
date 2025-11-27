#ifndef LAYER_HPP
#define LAYER_HPP

#include "../Matrix.hpp"

template <typename T>
class Layer {
public:
    virtual Matrix<T> Forward(const Matrix<T> X) = 0;
    virtual Matrix<T> Backward(const Matrix<T> grad) = 0;
};

#endif