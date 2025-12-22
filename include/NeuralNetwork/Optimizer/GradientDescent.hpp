#ifndef GRADIENTDESCENT_HPP
#define GRADIENTDESCENT_HPP

#include "Optimizer.hpp"

#include <memory>

template <typename T>
class GradientDescent {
private:
    T learning_rate_;
    Matrix<T> gradient_;

public:
    GradientDescent(std::shared_ptr<Matrix_Solver<T>> solver, T learning_rate):
        Optimizer<T>(solver), learning_rate_(learning_rate) {
    }

    void SetLearningRate(T lr) {
        this->learning_rate_ = lr;
    }

    void SetGradient(Matrix<T>& gradient) {
        this->gradient_ = gradient;
    }

    void Optimize(Matrix<T>& weights) {
        // TODO: implementation
    }
};

#endif