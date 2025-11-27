#ifndef DENSE_HPP
#define DENSE_HPP

#include "Layer.hpp"

template <typename T>
class Dense : public Layer<T> {
private:
    int in, out;
    Matrix<T> weights;
public:
    Dense(int, int);
    Dense(int, int, const Matrix<T>);

    Matrix<T> Forward(const Matrix<T>);
    Matrix<T> Backward(const Matrix<T>);
};

#endif