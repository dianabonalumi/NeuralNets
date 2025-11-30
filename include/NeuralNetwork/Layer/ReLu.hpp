#ifndef RELU_HPP
#define RELU_HPP

#include "Layer.hpp"
#include <algorithm> 

template <typename T>
class ReLU : public Layer<T> {
private:
    Matrix<T> lastInput; // Cache for backward pass

public:
    // Constructor
    ReLU(std::shared_ptr<Matrix_Solver<T>> solver) : Layer<T>(solver) {}

    // Forward Pass: f(x) = max(0, x)
    Matrix<T> Forward(const Matrix<T> X) override {
        lastInput = X;
        
        size_t r = X.rows();
        size_t c = X.cols();
        Matrix<T> output(r, c);

        const T* inData = X.Flatten();
        T* outData = output.Flatten();
        size_t size = r * c;

        for(size_t i = 0; i < size; ++i) {
            // Logic: Set negative values to zero
            outData[i] = std::max(static_cast<T>(0), inData[i]);
        }
        return output;
    }

    // Backward Pass: Derivative is 1 if x > 0, else 0
    Matrix<T> Backward(const Matrix<T> grad, T learning_rate) override {
        size_t r = grad.rows();
        size_t c = grad.cols();
        Matrix<T> inputGrad(r, c);

        const T* gradData = grad.Flatten();
        const T* inData = lastInput.Flatten();
        T* resultData = inputGrad.Flatten();
        size_t size = r * c;

        for(size_t i = 0; i < size; ++i) {
            if (inData[i] > 0) {
                resultData[i] = gradData[i]; // Pass through
            } else {
                resultData[i] = 0;           // Block gradient
            }
        }
        return inputGrad;
    }
};

#endif