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
    // Costruttore
    MSE(std::shared_ptr<Matrix_Solver<T>> solver) : Loss<T>(solver) {}



    Matrix<T> Compute(const Matrix<T> prediction, const Matrix<T> target) override {
        
        this->lastX = prediction;
        this->lastY = target;
        
        size_t rows = prediction.getRows();
        size_t cols = prediction.getCols();

        T sum_sq = 0;

        // 2. Ciclo di calcolo
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                
                T diff = prediction.Get(i, j) - target.Get(i, j);
                sum_sq += diff * diff;
            }
        }

        Formula: 0.5 * Sum Squared Error
        T loss_val = 0.5 * sum_sq;

       
        Matrix<T> result(1, 1);
        result.Set(0, 0, loss_val);

        return result;
    }
    Matrix<T> Gradient() {
        // Computation of MSE gradient
        return Matrix<T>(2,2);
    }
};

#endif