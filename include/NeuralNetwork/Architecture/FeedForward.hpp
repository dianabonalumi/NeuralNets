#ifndef FEEDFORWARD_HPP
#define FEEDFORWARD_HPP

#include "Architecture.hpp"
#include "../Layer/Layer.hpp"
#include "../Loss/Loss.hpp"
#include "../Matrix.hpp"

#include <vector>
#include <memory>

template <typename T>
class FeedForward : public Architecture<T> {
private:
    std::vector<std::shared_ptr<Layer<T>>> layers_;
    std::shared_ptr<Loss<T>> loss_;

public:
    FeedForward(const std::vector<std::shared_ptr<Layer<T>>> layers, const std::shared_ptr<Loss<T>> loss) 
        : layers_(layers), loss_(loss) {
    }

    Matrix<T> Train(const Matrix<T>& X, const Matrix<T>& Y, T learning_rate) {
        Matrix<T> out, grad;

        // forward pass
        out = Eval(X, Y);

        // backward pass
        grad = this->loss_->Gradient();
        for(auto it = this->layers_.rbegin();it != this->layers_.rend();it++) {
            grad = (*it)->Backward(grad, learning_rate);
        }

        return out;
    }

    Matrix<T> Eval(const Matrix<T>& X, const Matrix<T>& Y) {
        Matrix<T> pred = this->Predict(X);

        return this->loss_->Compute(pred, Y);
    }

    Matrix<T> Predict(const Matrix<T>& X) {
        Matrix<T> out = X;

        for(auto it = this->layers_.begin();it != this->layers_.end();it++) {
            out = (*it)->Forward(out);
        }

        return out;
    }
};

#endif