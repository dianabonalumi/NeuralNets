#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "Layer.hpp"
#include <cmath> 

template <typename T>
class Sigmoid : public Layer<T> {
private:
    Matrix<T> lastOutput; // Cache for backward pass

public:
    Sigmoid(std::shared_ptr<Matrix_Solver<T>> solver): Layer<T>(solver) {}

    Matrix<T> Forward(const Matrix<T> X) override {
        size_t r = X.rows();
        size_t c = X.cols();
        Matrix<T> output(r, c);

        const T* inData = X.Flatten();
        T* outData = output.Flatten();
        size_t size = r * c;

        for(size_t i = 0; i < size; ++i) {
            // f(x) = 1 / (1 + e^-x)
            outData[i] = 1.0 / (1.0 + std::exp(-inData[i]));
        }

        lastOutput = output; 
        return output;
    }

    Matrix<T> Backward(const Matrix<T> grad, T learning_rate) override {
        size_t r = grad.rows();
        size_t c = grad.cols();
        Matrix<T> inputGrad(r, c);

        const T* gradData = grad.Flatten();
        const T* outData = lastOutput.Flatten();
        T* resultData = inputGrad.Flatten();
        size_t size = r * c;

        for(size_t i = 0; i < size; ++i) {
            T sig = outData[i];
            // Derivative: f'(x) = f(x) * (1 - f(x))
            resultData[i] = gradData[i] * (sig * (1.0 - sig));
        }

        return inputGrad;
    }
};

#endif