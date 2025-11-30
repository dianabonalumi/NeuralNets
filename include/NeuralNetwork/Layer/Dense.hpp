#ifndef DENSE_HPP
#define DENSE_HPP

#include "Layer.hpp"
#include <random>
#include <iostream>

template <typename T>
class Dense : public Layer<T> {
private:
    int in;
    int out;
    Matrix<T> weights;
    Matrix<T> lastInput; // Cache for Backward pass

    // Initialize weights with small random values to break symmetry
    void initWeights() {
         std::default_random_engine generator;
         std::uniform_real_distribution<T> distribution(-0.1, 0.1);
         
         T* wData = weights.Flatten();
         size_t total_weights = in * out;
         for(size_t i = 0; i < total_weights; ++i) {
             wData[i] = distribution(generator);
         }
    }

public:
    // Constructor: Random initialization
    Dense(std::shared_ptr<Matrix_Solver<T>> solver, int in_features, int out_features)
        : Layer<T>(solver), in(in_features), out(out_features), weights(in_features, out_features) {
            initWeights();
    }
    
    // Constructor: Manual weights (useful for debugging/loading weights)
    Dense(std::shared_ptr<Matrix_Solver<T>> solver, int in_features, int out_features, const Matrix<T> w) 
        : Layer<T>(solver), in(in_features), out(out_features), weights(w) {
    }

    // --- FORWARD PASS ---
    // Computes Y = X * W
    Matrix<T> Forward(const Matrix<T> X) override {
        // Dimension check: Input features must match layer input size
        if (X.cols() != (size_t)in) {
            std::cerr << "Dense Error: Input dim " << X.cols() << " != Layer in " << in << std::endl;
            throw std::runtime_error("Dimension mismatch in Dense Forward");
        }

        lastInput = X; // Save input for backward pass
        
        size_t batchSize = X.rows();
        Matrix<T> Y(batchSize, out); 

        // Perform Matrix Multiplication using the Solver
        this->solver_->multiply(batchSize, out, in, X.Flatten(), weights.Flatten(), Y.Flatten());
        
        return Y;
    }

    // --- BACKWARD PASS ---
    // Computes dL/dX (to pass to previous layer) and dL/dW (for weight update)
    Matrix<T> Backward(const Matrix<T> grad, T learning_rate) override {
        size_t batchSize = grad.rows();
        
        // 1. Compute Gradient w.r.t Input (dL/dX) -> passes to previous layer
        // Formula: dX = dY * W^T
        Matrix<T> W_T = weights.Transpose(); 
        Matrix<T> inputGrad(batchSize, in);
        this->solver_->multiply(batchSize, in, out, grad.Flatten(), W_T.Flatten(), inputGrad.Flatten());

        // 2. Compute Gradient w.r.t Weights (dL/dW) -> used for update
        // Formula: dW = X^T * dY
        Matrix<T> X_T = lastInput.Transpose();
        Matrix<T> weightGrad(in, out);
        this->solver_->multiply(in, out, batchSize, X_T.Flatten(), grad.Flatten(), weightGrad.Flatten());

        // 3. Update Weights (Gradient Descent)
        // Formula: W = W - learning_rate * dW
        T* wData = weights.Flatten();
        T* gData = weightGrad.Flatten();
        size_t total_weights = in * out;

        for(size_t i = 0; i < total_weights; ++i) {
            wData[i] -= learning_rate * gData[i];
        }

        return inputGrad;
    }
    
    // Getter for testing/debugging
    const Matrix<T>& getWeights() const { return weights; }
};

#endif