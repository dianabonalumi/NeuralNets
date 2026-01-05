#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "Architecture.hpp"
#include "../Layer/Layer.hpp"
#include "../Loss/Loss.hpp"
#include "../Matrix.hpp"
#include "../../matrix_solver.hpp"
#include "../Optimizer/Optimizer.hpp"

#include <vector>
#include <memory>

template <typename T>
class Encoder : public Architecture<T> {
private:
    const std::shared_ptr<Matrix_Solver<T>> solver_;
    const std::shared_ptr<Optimizer<T>> optimizer_;
    const std::shared_ptr<Loss<T>> loss_;
    const int window_, stride_;
    const int in_shape_, out_shape_;

public:
    Encoder(const std::shared_ptr<Matrix_Solver<T>>& solver,
        const std::shared_ptr<Optimizer<T>>& optimizer, 
        const std::shared_ptr<Loss<T>>& loss,
        const int& window,
        const int& stride,
        const int& in_shape,
        const int& out_shape) 
        : solver_(solver), optimizer_(optimizer), loss_(loss), in_shape_(in_shape), out_shape_(out_shape),
        window_(window), stride_(stride) {
    }

    Matrix<T> Train(const Matrix<T>& X, const Matrix<T>& Y) {
    }

    Matrix<T> Eval(const Matrix<T>& X, const Matrix<T>& Y) {
    }

    Matrix<T> Predict(const Matrix<T>& X) {
    }

    Matrix<T> Backward(const Matrix<T>& X, const Matrix<T>& grad) {
        
    }
};

#endif