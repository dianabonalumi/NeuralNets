#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "Layer.hpp"
#include <cmath> 

template <typename T>
class Softmax : public Layer<T> {
private:
    Matrix<T> lastOutput;

public:
    Softmax(const std::shared_ptr<Matrix_Solver<T>>& solver) : Layer<T>(solver) {}

    Matrix<T> Forward(const Matrix<T>& X) override {
        size_t rows = X.rows();
        size_t cols = X.cols();
        Matrix<T> output(rows, cols);

        for (size_t i = 0; i < rows; ++i) {
            T maxVal = X.Get(i, 0); // Per stabilità numerica
            for (size_t j = 1; j < cols; ++j) if (X.Get(i, j) > maxVal) maxVal = X.Get(i, j);

            T sum = 0;
            for (size_t j = 0; j < cols; ++j) {
                T val = std::exp(X.Get(i, j) - maxVal);
                output.Set(i, j, val);
                sum += val;
            }
            for (size_t j = 0; j < cols; ++j) output.Set(i, j, output.Get(i, j) / sum);
        }
        lastOutput = output;
        return output;
    }

    Matrix<T> Backward(const Matrix<T>& grad) override {
        // Nota: Spesso Softmax + CrossEntropy si semplifica in (pred - target)
        // Se implementata standalone, la derivata è più complessa (Jacobiana).
        return grad; // Implementazione semplificata per uso combinato con Loss
    }

    void WeightInitialization(const WeightInit& technique) override {}
};

#endif // SOFTMAX_HPP