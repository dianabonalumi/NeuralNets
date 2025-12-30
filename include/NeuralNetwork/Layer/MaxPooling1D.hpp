#ifndef MAXPOOLING1D_HPP
#define MAXPOOLING1D_HPP

#include "Layer.hpp"
#include <algorithm>
#include <vector>

template<typename T>
class MaxPooling1D : public Layer<T> {
private:
    const int pool_size_;
    const int stride_;
    const int in_channels_;
    const int length_;
    
    // Store indices of max values for backward pass
    std::vector<std::vector<int>> max_indices_;
    Matrix<T> lastInput;

public:
    // Constructor
    MaxPooling1D(const std::shared_ptr<Matrix_Solver<T>>& solver,
                 const int& pool_size, const int& stride, 
                 const int& in_channels, const int& length):
        Layer<T>(solver, nullptr), pool_size_(pool_size), stride_(stride),
        in_channels_(in_channels), length_(length) {}

    // Forward Pass: Apply max pooling over input
    Matrix<T> Forward(const Matrix<T>& X) override {
        lastInput = X;
        size_t batchSize = X.rows();
        
        // Calculate output length after pooling
        int output_length = (length_ - pool_size_) / stride_ + 1;
        
        Matrix<T> Y(batchSize, in_channels_ * output_length);
        
        // Clear and resize indices storage
        max_indices_.clear();
        max_indices_.resize(batchSize);
        
        const T* inData = X.Flatten();
        T* outData = Y.Flatten();
        
        for (size_t b = 0; b < batchSize; ++b) {
            max_indices_[b].resize(in_channels_ * output_length);
            
            for (int c = 0; c < in_channels_; ++c) {
                for (int out_pos = 0; out_pos < output_length; ++out_pos) {
                    // Get the window for this pooling operation
                    int in_start = out_pos * stride_;
                    int in_end = in_start + pool_size_;
                    
                    T max_val = std::numeric_limits<T>::lowest();
                    int max_idx = -1;
                    
                    // Find max value in the pooling window
                    for (int in_pos = in_start; in_pos < in_end && in_pos < length_; ++in_pos) {
                        int src_idx = b * X.cols() + c * length_ + in_pos;
                        if (inData[src_idx] > max_val) {
                            max_val = inData[src_idx];
                            max_idx = in_pos;
                        }
                    }
                    
                    // Store output and the index of max value
                    int out_idx = b * Y.cols() + c * output_length + out_pos;
                    outData[out_idx] = max_val;
                    max_indices_[b][c * output_length + out_pos] = max_idx;
                }
            }
        }
        
        return Y;
    }

    // Backward Pass: Route gradients only to max elements
    Matrix<T> Backward(const Matrix<T>& grad) override {
        size_t batchSize = grad.rows();
        int output_length = (length_ - pool_size_) / stride_ + 1;
        
        Matrix<T> inputGrad(batchSize, lastInput.cols());
        
        T* inGradData = inputGrad.Flatten();
        const T* gradData = grad.Flatten();
        
        // Initialize to zero
        std::fill(inGradData, inGradData + inputGrad.rows() * inputGrad.cols(), static_cast<T>(0));
        
        // Route gradient back to the position of max values
        for (size_t b = 0; b < batchSize; ++b) {
            for (int c = 0; c < in_channels_; ++c) {
                for (int out_pos = 0; out_pos < output_length; ++out_pos) {
                    int grad_idx = b * grad.cols() + c * output_length + out_pos;
                    int max_idx = max_indices_[b][c * output_length + out_pos];
                    
                    // Route gradient to the input position that had the max value
                    int in_grad_idx = b * lastInput.cols() + c * length_ + max_idx;
                    inGradData[in_grad_idx] = gradData[grad_idx];
                }
            }
        }
        
        return inputGrad;
    }

    void WeightInitialization(const WeightInit& technique) override {} // No weights in pooling layer
};

#endif
