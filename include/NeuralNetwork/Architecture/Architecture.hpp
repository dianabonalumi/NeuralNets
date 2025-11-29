#ifndef ARCHITECTURE_HPP
#define ARCHITECTURE_HPP

#include "../Matrix.hpp"

template<typename T>
class Architecture {
public:
    virtual Matrix<T> Train(const Matrix<T>&, const Matrix<T>&, T) = 0;
    virtual Matrix<T> Eval(const Matrix<T>&, const Matrix<T>&) = 0;
    virtual Matrix<T> Predict(const Matrix<T>&) = 0;
};

#endif