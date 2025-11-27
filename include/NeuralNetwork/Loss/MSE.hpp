#ifndef MSE_HPP
#define MSE_HPP

#include "Loss.hpp"

template <typename T>
class MSE : public Loss<T> {
private:
    Matrix<T> lastX;

public:
    Matrix<T> Compute(const Matrix<T> X) {
        // Computation of MSE loss
        return X;
    }
    Matrix<T> Gradient() {
        // Computation of MSE gradient
        return Matrix<T>(2,2);
    }
};

#endif