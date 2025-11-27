#include "../../../include/NeuralNetwork/Layer/Sigmoid.hpp"

template <typename T>
Matrix<T> Sigmoid<T>::Forward(const Matrix<T> X) {
    // Sigmoid forward pass
    Matrix<T> K = X;
    K[0] -= 1;
    return K;
}

template <typename T>
Matrix<T> Sigmoid<T>::Backward(const Matrix<T> grad) {
    // Sigmoid backward pass
    return grad;
}

template class Sigmoid<double>;
template class Sigmoid<float>;