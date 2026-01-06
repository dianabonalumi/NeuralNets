#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include "Architecture.hpp"

#include "Encoder.hpp"
#include "Decoder.hpp"

template<typename T>
class Autoencoder: public Architecture<T> {
private:
    Encoder<T> enc;
    Decoder<T> dec;

public:
    Autoencoder<T>(const std::shared_ptr<Matrix_Solver<T>>& solver,
        const std::shared_ptr<Optimizer<T>>& optimizer,
        int in_shape, int bottleneck_shape,
        int window_size, int stride) {
            this->enc = Encoder<T>(solver, optimizer, window_size, stride, in_shape, bottleneck_shape);
            this->dec = Decoder<T>(solver, optimizer, bottleneck_shape, in_shape);
    }

    Matrix<T> Predict(const Matrix<T>& X) {
        Matrix<T> encoded = this->enc.Predict(X);

        return this->dec.Predict(encoded);
    }

    Matrix<T> Backward(const Matrix<T>& grad) {
        Matrix<T> decGrad = this->dec.Backward(X, grad);

        return this->enc.Backward(X, decGrad);
    }

    Encoder<T> GetEncoder() {
        return this->enc;
    }

    Decoder<T> GetDecoder() {
        return this->dec;
    }
};

#endif