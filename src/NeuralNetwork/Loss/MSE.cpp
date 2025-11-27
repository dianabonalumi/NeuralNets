#include "../../../include/NeuralNetwork/Loss/MSE.hpp"

template <typename T>
Matrix<T> MSE<T>::Compute(const Matrix<T> X) {
    // Computation of MSE loss
    return X;
}

template <typename T>
Matrix<T> MSE<T>::Gradient() {
    // Computation of MSE gradient
    return Matrix<T>(2,2);
}

template class MSE<double>;
template class MSE<float>;