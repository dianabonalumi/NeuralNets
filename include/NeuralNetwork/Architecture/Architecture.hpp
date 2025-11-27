#ifndef ARCHITECTURE_HPP
#define ARCHITECTURE_HPP

#include "../Matrix.hpp"

template<typename T>
class Architecture {
public:
    virtual void Train(const Matrix<T>) = 0;
    virtual Matrix<T> Eval(const Matrix<T>) = 0;
};

#endif