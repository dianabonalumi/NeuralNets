#ifndef DECODER_HPP
#define DECODER_HPP

#include "Architecture.hpp"
#include "../Layer/Layer.hpp"
#include "../Loss/Loss.hpp"
#include "../Matrix.hpp"
#include "../../matrix_solver.hpp"
#include "../Optimizer/Optimizer.hpp"

#include <vector>
#include <memory>

template <typename T>
class Decoder : public Architecture<T> {
private:
    const std::shared_ptr<Matrix_Solver<T>> solver_;
    const std::shared_ptr<Optimizer<T>> optimizer_;
    const std::shared_ptr<Loss<T>> loss_;
    const int in_shape_, out_shape_;

public:
    Decoder(const std::shared_ptr<Matrix_Solver<T>>& solver,
        const std::shared_ptr<Optimizer<T>>& optimizer, 
        const std::shared_ptr<Loss<T>>& loss,
        const int& in_shape,
        const int& out_shape) 
        : solver_(solver), optimizer_(optimizer), loss_(loss), in_shape_(in_shape), out_shape_(out_shape) {
    }

    Matrix<T> Train(const Matrix<T>& X, const Matrix<T>& Y) {
    }

    Matrix<T> Eval(const Matrix<T>& X, const Matrix<T>& Y) {
    }

    Matrix<T> Predict(const Matrix<T>& X) {
    }

    Matrix<T> GetGradient() {
        
    }

    void SetGradient(Matrix<T>& gradient) {
        
    }
};

#endif