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
    std::vector<std::shared_ptr<Layer<T>>> layers;
    std::shared_ptr<Loss<T>> loss;

public:
    FeedForward(const std::vector<std::shared_ptr<Layer<T>>>, const std::shared_ptr<Loss<T>>);

    void Train(const Matrix<T>);
    const Matrix<T> Eval(const Matrix<T>);
};

#endif