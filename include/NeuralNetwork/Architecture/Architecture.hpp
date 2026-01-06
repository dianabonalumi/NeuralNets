#ifndef ARCHITECTURE_HPP
#define ARCHITECTURE_HPP

#include "../Matrix.hpp"

template<typename T>
class Architecture {
public:
    virtual Matrix<T> Train(const Matrix<T>&, const Matrix<T>&) { return Matrix<T>(); };     // deprecated
    virtual Matrix<T> Eval(const Matrix<T>&, const Matrix<T>&) { return Matrix<T>(); };      // deprecated
    virtual Matrix<T> Eval() { return Matrix<T>(); };                                        // computes and returns loss value
    virtual Matrix<T> Predict(const Matrix<T>&) = 0;
    virtual Matrix<T> Backward(const Matrix<T>&, const Matrix<T>&) = 0;
};

#endif