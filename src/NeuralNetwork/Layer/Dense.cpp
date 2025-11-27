#include "../../../include/NeuralNetwork/Layer/Dense.hpp"

template <typename T>
Dense<T>::Dense(int inp, int out) {
    // Dense constructor (without weights)
}

template <typename T>
Dense<T>::Dense(int inp, int out, const Matrix<T> weights) {
    // Dense constructor (with weights)
}

template <typename T>
Matrix<T> Dense<T>::Forward(const Matrix<T> X) {
    // Dense forward pass
    return X;
}

template <typename T>
Matrix<T> Dense<T>::Backward(const Matrix<T> grad) {
    // Dense backward pass
    return grad;
}

template class Dense<double>;
template class Dense<float>;