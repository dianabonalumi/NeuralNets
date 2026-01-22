#ifndef GRADIENTDESCENT_HPP
#define GRADIENTDESCENT_HPP

#include "Optimizer.hpp"

template <typename T>
class GradientDescent : public Optimizer<T> {
private:
    const T& learning_rate_;

public:
    GradientDescent(const std::shared_ptr<Matrix_Solver<T>> solver, const T& learning_rate):
        Optimizer<T>(solver), learning_rate_(learning_rate) {}

    void Optimize(Matrix<T>& weights, const Matrix<T>& gradient) override {
        T* wData = weights.Flatten();
        const T* gData = gradient.Flatten();

        size_t total_weights = weights.rows() * weights.cols();

        // could be optimized using the solver
        for(size_t i = 0; i < total_weights; ++i) {
            wData[i] -= this->learning_rate_ * gData[i];
        }
    }
};

#endif