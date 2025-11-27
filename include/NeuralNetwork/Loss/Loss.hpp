#ifndef LOSS_HPP
#define LOSS_HPP

#include "../Matrix.hpp"

template <typename T>
class Loss {
public:
    virtual Matrix<T> Compute(const Matrix<T>) = 0;
    virtual Matrix<T> Gradient() = 0;
};

#endif