#ifndef LAYER_HPP
#define LAYER_HPP

#include "../Matrix.hpp"
#include "../../matrix_solver.hpp"

#include <memory>

template <typename T>
class Layer {
protected:
    std::shared_ptr<Matrix_Solver<T>> solver_; 
public:
    Layer(std::shared_ptr<Matrix_Solver<T>> solver): solver_(solver) {}
    
    // Virtual destructor: ensures proper cleanup of derived classes
    virtual ~Layer() = default;

    virtual Matrix<T> Forward(const Matrix<T> X) = 0;
    
    // Updated Backward signature to support Gradient Descent
    virtual Matrix<T> Backward(const Matrix<T> grad, T learning_rate) = 0;
};

#endif