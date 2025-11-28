#ifndef MSE_HPP
#define MSE_HPP

#include "Loss.hpp"

#include <memory>

template <typename T>
class MSE : public Loss<T> {
private:
    Matrix<T> lastX;
    Matrix<T> lastY;

public:
    MSE(std::shared_ptr<Matrix_Solver<T>> solver): Loss<T>(solver) {}

    Matrix<T> Compute(const Matrix<T> X, const Matrix<T> Y) {
        // Computation of MSE loss
        return X;
    }
    Matrix<T> Gradient() {
        // Computation of MSE gradient
        return Matrix<T>(2,2);
    }
};

#endif