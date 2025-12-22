#ifndef CROSS_ENTROPY_HPP
#define CROSS_ENTROPY_HPP

#include "Loss.hpp"
#include <memory>

template <typename T>
class CrossEntropy : public Loss<T> {
private:
    Matrix<T> lastX;
    Matrix<T> lastY;

public:
    // Costruttore
    CrossEntropy(std::shared_ptr<Matrix_Solver<T>> solver) : Loss<T>(solver) {}

    Matrix<T> Compute(const Matrix<T> prediction, const Matrix<T> target) override {
        // TODO: implementation
    }

    Matrix<T> Gradient() override {
        // TODO: implementation
    }
};

#endif