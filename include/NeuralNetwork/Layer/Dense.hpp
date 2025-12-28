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
    Matrix<T> lastInput; 

public:
    // Costruttore con Solver e Optimizer (tramite Layer)
    Dense(const std::shared_ptr<Matrix_Solver<T>>& solver, 
          const std::shared_ptr<Optimizer<T>>& optimizer, 
          int in_features, int out_features)
        : Layer<T>(solver, optimizer), in(in_features), out(out_features), weights(in_features, out_features) {
    }

    // Inizializzazione pesi
    void WeightInitialization(const WeightInit& technique) override {
        size_t total_weights = (size_t)in * out;
        T* wData = weights.Flatten();
        
        std::random_device rd;
        std::default_random_engine generator(rd());
        T std_dev;

        if (technique == WeightInit::Xavier) {
            std_dev = std::sqrt(static_cast<T>(2.0) / (in + out));
        } else { 
            std_dev = std::sqrt(static_cast<T>(2.0) / in);
        }

        std::normal_distribution<T> distribution(static_cast<T>(0.0), std_dev);
        for(size_t i = 0; i < total_weights; ++i) {
            wData[i] = distribution(generator);
        }
    }

    Matrix<T> Forward(const Matrix<T>& X) override {
        if (X.cols() != (size_t)in) {
            throw std::runtime_error("Dimension mismatch in Dense Forward");
        }

        lastInput = X; 
        size_t batchSize = X.rows();
        Matrix<T> Y(batchSize, out); 

        this->solver_->multiply(batchSize, out, in, X.Flatten(), weights.Flatten(), Y.Flatten());
        return Y;
    }

    // CORRETTO: Aggiunta la & a grad per matchare la classe base
    Matrix<T> Backward(const Matrix<T>& grad) override {
        size_t batchSize = grad.rows();
        
        // 1. dL/dX = grad * W^T
        Matrix<T> W_T = weights.Transpose(); 
        Matrix<T> inputGrad(batchSize, in);
        this->solver_->multiply(batchSize, in, out, grad.Flatten(), W_T.Flatten(), inputGrad.Flatten());

        // 2. dL/dW = X^T * grad
        Matrix<T> X_T = lastInput.Transpose();
        Matrix<T> weightGrad(in, out);
        this->solver_->multiply(in, out, batchSize, X_T.Flatten(), grad.Flatten(), weightGrad.Flatten());

        // 3. Update Weights tramite l'Optimizer del Layer
        if (this->optimizer_) {
            this->optimizer_->Optimize(this->weights, weightGrad);
        }

        return inputGrad;
    }
    
    const Matrix<T>& getWeights() const { return weights; }
};

#endif