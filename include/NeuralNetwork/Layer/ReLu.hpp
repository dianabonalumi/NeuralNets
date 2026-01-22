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
    ReLU(const std::shared_ptr<Matrix_Solver<T>>& solver) : Layer<T>(solver) {}

    // Forward Pass: f(x) = max(0, x)
    Matrix<T> Forward(const Matrix<T>& X) override {
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

    Matrix<T> Backward(const Matrix<T>& grad) override { // Rimosso learning_rate
        size_t r = grad.rows();
        size_t c = grad.cols();
        Matrix<T> inputGrad(r, c);

        const T* gradData = grad.Flatten();
        const T* inData = lastInput.Flatten();
        T* resultData = inputGrad.Flatten();
        size_t size = r * c;

        for(size_t i = 0; i < size; ++i) {
            // Derivata: 1 se x > 0, altrimenti 0
            resultData[i] = (inData[i] > 0) ? gradData[i] : static_cast<T>(0);
        }
        return inputGrad;
    }

    void WeightInitialization(const WeightInit& technique) override {} // Layer senza pesi
};

#endif