#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include "Architecture.hpp"

#include "../Loss/Loss.hpp"

#include <string>
#include <fstream>

#include "Encoder.hpp"
#include "Decoder.hpp"

template<typename T>
class Autoencoder: public Architecture<T> {
private:
    Encoder<T> enc;
    Decoder<T> dec;
    const std::shared_ptr<Loss<T>> loss_;

    Matrix<T> pred, grad;

public:
    Autoencoder<T>(const std::shared_ptr<Matrix_Solver<T>>& solver,
        const std::shared_ptr<Optimizer<T>>& optimizer,
        const std::shared_ptr<Loss<T>>& loss,
        const int in_shape, const int bottleneck_shape, const int hidden_state,
        const int window_size, const int stride): loss_(loss), 
        enc(solver, optimizer, window_size, stride, in_shape, bottleneck_shape, hidden_state),
        dec(solver, optimizer, bottleneck_shape, in_shape) {
        }

    Matrix<T> Predict(const Matrix<T>& X) {
        Matrix<T> encoded = this->enc.Predict(X);

        this->pred = this->dec.Predict(encoded);

        return this->pred;
    }

    Matrix<T> Eval(const Matrix<T>& target) {
        Matrix<T> result = this->loss_->Compute(this->pred, target);

        this->grad = this->loss_->Gradient();

        return result;
    }

    Matrix<T> Backward() {
        this->dec.SetGradient(this->grad);

        Matrix<T> grad = this->dec.Backward();

        this->enc.SetGradient(grad);
        grad = this->enc.Backward();

        return grad;
    }

    Encoder<T> GetEncoder() {
        return this->enc;
    }

    Decoder<T> GetDecoder() {
        return this->dec;
    }
};

#endif