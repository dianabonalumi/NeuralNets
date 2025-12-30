#ifndef CNN_HPP
#define CNN_HPP

#include "Layer.hpp"

#include <random>

template<typename T>
class CNN1D: public Layer<T> {
private:
    const int in_channels_;
    const int out_channels_;
    const int kernel_size_;
    const int length_;

    Matrix<T> weights;
    Matrix<T> lastInput;

public:
    // input is passed as a vector => has to be reshaped to correctly apply the weights
    CNN1D(const std::shared_ptr<Matrix_Solver<T>>& solver, 
        const std::shared_ptr<Optimizer<T>>& optimizer,
        const int& in_channels, const int& out_channels,
        const int& kernel_size, const int& length):
        Layer<T>(solver, optimizer), in_channels_(in_channels), out_channels_(out_channels), 
        kernel_size_(kernel_size), length_(length), weights(in_channels * kernel_size, out_channels) {
        }

    void WeightInitialization(const WeightInit& technique) override {
        size_t total_weights = (size_t)(out_channels_ * kernel_size_ * in_channels_);
        T* wData = weights.Flatten();
        
        std::random_device rd;
        std::default_random_engine generator(rd());
        T std_dev;

        if (technique == WeightInit::Xavier) {
            std_dev = std::sqrt(static_cast<T>(2.0) / (out_channels_ + kernel_size_ * in_channels_));
        } else { 
            std_dev = std::sqrt(static_cast<T>(2.0) / kernel_size_ * in_channels_);
        }

        std::normal_distribution<T> distribution(static_cast<T>(0.0), std_dev);
        for(size_t i = 0; i < total_weights; ++i) {
            wData[i] = distribution(generator);
        }
    }

    void Convolve(T* data, T* wData, T* out, int padding) {
        for(int i = 0;i < length_;i++) {
            this->solver_->multiply(1, out_channels_, in_channels_ * kernel_size_, data, wData, out);
            data += in_channels_;
            out += out_channels_;
        }
    }

    Matrix<T> AddPadding(const Matrix<T>& X, int padding) {
        Matrix<T> padX(X.rows(), X.cols() + 2 * padding * in_channels_);

        for(int i = 0;i < X.rows();i++) {
            for(int j = 0;j < X.cols();j++) {   // for each series in the batch
                padX.Set(i, j + padding * in_channels_, X.Get(i, j));
            }
        }

        return padX;
    }

    Matrix<T> Forward(const Matrix<T>& X) override {
        lastInput = X; 
        size_t batchSize = X.rows();

        Matrix<T> Y(batchSize, out_channels_ * length_);

        int padding = kernel_size_ / 2;
        Matrix<T> padX = AddPadding(X, padding);

        T* data = padX.Flatten();
        T* wData = weights.Flatten();
        T* out = Y.Flatten();

        for(int i = 0;i < batchSize;i++) {
            Convolve(data, wData, out, padding);

            data += padX.cols();
            out += Y.cols();
        }

        return Y;
    }

    Matrix<T> Backward(const Matrix<T>& grad) override {
        size_t batchSize = grad.rows();
        int padding = kernel_size_ / 2;
        
        // Pad input for backward pass
        Matrix<T> padX = AddPadding(lastInput, padding);
        
        // Initialize weight gradient with same shape as weights
        Matrix<T> weightGrad(in_channels_ * kernel_size_, out_channels_);
        T* wGradData = weightGrad.Flatten();
        std::fill(wGradData, wGradData + weightGrad.rows() * weightGrad.cols(), static_cast<T>(0));
        
        T* padX_data = padX.Flatten();
        const T* grad_data = grad.Flatten();
        T* wData = weights.Flatten();
        
        // Compute weight gradients: accumulate outer products of patches and gradients
        for (size_t b = 0; b < batchSize; ++b) {
            for (int t = 0; t < length_; ++t) {
                // Pointer to patch at time position t (shape: in_channels_ * kernel_size_)
                T* patch_ptr = padX_data + b * padX.cols() + t * in_channels_;
                
                // Gradient at this time position (shape: out_channels_)
                const T* grad_pos = grad_data + b * grad.cols() + t * out_channels_;
                
                // Accumulate outer product: wGrad += patch @ grad^T
                for (int i = 0; i < in_channels_ * kernel_size_; ++i) {
                    for (int o = 0; o < out_channels_; ++o) {
                        wGradData[i * out_channels_ + o] += patch_ptr[i] * grad_pos[o];
                    }
                }
            }
        }
        
        // Compute input gradients for padded input: padGrad = grad @ W^T
        Matrix<T> paddedInputGrad(padX.rows(), padX.cols());
        T* padGradData = paddedInputGrad.Flatten();
        std::fill(padGradData, padGradData + paddedInputGrad.rows() * paddedInputGrad.cols(), static_cast<T>(0));
        
        for (size_t b = 0; b < batchSize; ++b) {
            for (int t = 0; t < length_; ++t) {
                // Gradient at this time position
                const T* grad_pos = grad_data + b * grad.cols() + t * out_channels_;
                
                // Output patch gradient destination
                T* patch_grad = padGradData + b * padX.cols() + t * in_channels_;
                
                // Compute and accumulate: patch_grad += grad @ W^T
                for (int k = 0; k < in_channels_ * kernel_size_; ++k) {
                    T val = 0;
                    for (int o = 0; o < out_channels_; ++o) {
                        val += grad_pos[o] * wData[k * out_channels_ + o];
                    }
                    patch_grad[k] += val;
                }
            }
        }
        
        // Remove padding from input gradient
        Matrix<T> inputGrad(batchSize, lastInput.cols());
        T* inGradData = inputGrad.Flatten();
        
        for (size_t b = 0; b < batchSize; ++b) {
            for (int t = 0; t < length_; ++t) {
                for (int c = 0; c < in_channels_; ++c) {
                    // Source index in padded gradient
                    int src_idx = b * padX.cols() + (t + padding) * in_channels_ + c;
                    
                    // Destination index in unpadded gradient
                    int dst_idx = b * lastInput.cols() + t * in_channels_ + c;
                    
                    inGradData[dst_idx] = padGradData[src_idx];
                }
            }
        }
        
        // Update weights using optimizer
        if (this->optimizer_) {
            this->optimizer_->Optimize(weights, weightGrad);
        }
        
        return inputGrad;
    }

    Matrix<T>& getWeights() {
        return this->weights;
    }
};

#endif