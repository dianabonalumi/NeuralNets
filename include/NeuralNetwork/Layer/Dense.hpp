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
    Matrix<T> bias;
    Matrix<T> lastInput; 

public:
    // Costruttore con Solver e Optimizer (tramite Layer)
    Dense(const std::shared_ptr<Matrix_Solver<T>>& solver, 
          const std::shared_ptr<Optimizer<T>>& optimizer, 
          int in_features, int out_features)
        : Layer<T>(solver, optimizer), in(in_features), out(out_features), weights(in_features, out_features), bias(1, out_features) {
    }

    // Inizializzazione pesi
    void WeightInitialization(const WeightInit& technique) override {
        size_t total_weights = (size_t)in * out;
        T* wData = weights.Flatten();
        T* bData = bias.Flatten();
        
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
        
        for(int i = 0; i < out; ++i) {
            bData[i] = static_cast<T>(0.0);
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
        
        // Add bias to each sample in the batch
        T* yData = Y.Flatten();
        const T* bData = bias.Flatten();
        for(size_t i = 0; i < batchSize; ++i) {
            for(int j = 0; j < out; ++j) {
                yData[i * out + j] += bData[j];
            }
        }
        
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

        // 3. dL/db = sum over batch dimension of grad
        Matrix<T> biasGrad(1, out);
        T* bGData = biasGrad.Flatten();
        const T* gData = grad.Flatten();
        for(int j = 0; j < out; ++j) {
            bGData[j] = static_cast<T>(0.0);
            for(size_t i = 0; i < batchSize; ++i) {
                bGData[j] += gData[i * out + j];
            }
        }

        // 4. Update Weights and Bias tramite l'Optimizer del Layer
        if (this->optimizer_) {
            this->optimizer_->Optimize(this->weights, weightGrad);
            this->optimizer_->Optimize(this->bias, biasGrad);
        }

        return inputGrad;
    }
    
    const Matrix<T>& getWeights() const { return weights; }

    T* Serialize() {
        // Serialize weights and bias: size + weights + bias
        size_t total_weights = (size_t)in * out;
        size_t total_bias = (size_t)out;
        T* buffer = new T[total_weights + total_bias + 2];
        buffer[0] = static_cast<T>(total_weights);
        
        const T* w_data = weights.Flatten();
        for(size_t i = 0; i < total_weights; ++i) {
            buffer[i + 1] = w_data[i];
        }
        
        buffer[total_weights + 1] = static_cast<T>(total_bias);
        const T* b_data = bias.Flatten();
        for(size_t i = 0; i < total_bias; ++i) {
            buffer[total_weights + 2 + i] = b_data[i];
        }
        
        return buffer;
    }

    void Deserialize(T* data) {
        if (!data) return;
        
        size_t total_weights = (size_t)in * out;
        T* w_data = weights.Flatten();
        
        for(size_t i = 0; i < total_weights; ++i) {
            w_data[i] = data[i + 1];
        }
        
        size_t total_bias = (size_t)out;
        T* b_data = bias.Flatten();
        
        for(size_t i = 0; i < total_bias; ++i) {
            b_data[i] = data[total_weights + 2 + i];
        }
    }
};

#endif