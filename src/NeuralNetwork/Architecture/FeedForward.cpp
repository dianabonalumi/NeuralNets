#include "../../../include/NeuralNetwork/Architecture/FeedForward.hpp"

template <typename T>
FeedForward<T>::FeedForward(const std::vector<std::shared_ptr<Layer<T>>> layers, const std::shared_ptr<Loss<T>> loss) : layers(layers), loss(loss){
}

template <typename T>
void FeedForward<T>::Train(const Matrix<T> X) {
    // Train implementation for FeedForward
}

template<typename T>
Matrix<T> FeedForward<T>::Eval(const Matrix<T> X) {
    // Eval implementation for FeedForward
    return X;
}

template class FeedForward<double>;
template class FeedForward<float>;