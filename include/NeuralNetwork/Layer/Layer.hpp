#ifndef LAYER_HPP
#define LAYER_HPP

#include "../Matrix.hpp"
#include "../Optimizer/Optimizer.hpp"
#include "../../matrix_solver.hpp"
#include "WeightInitialization.hpp"

#include <memory>

template <typename T>
class Layer {
protected:
    const std::shared_ptr<Matrix_Solver<T>> solver_; 
    const Optimizer<T> optimizer_;
public:
    // DEPRECATED
    Layer(const std::shared_ptr<Matrix_Solver<T>>& solver): solver_(solver), optimizer_(nullptr) {}

    // New constructor for general Optimizer
    Layer(const std::shared_ptr<Matrix_Solver<T>>& solver, const std::shared_ptr<Optimizer<T>>& optimizer):
        solver_(solver), optimizer_(optimizer) {}
    
    // Virtual destructor: ensures proper cleanup of derived classes
    virtual ~Layer() = default;

    virtual Matrix<T> Forward(const Matrix<T> X) = 0;
    
    // DEPRECATED
    // Updated Backward signature to support Gradient Descent
    virtual Matrix<T> Backward(const Matrix<T> grad, T learning_rate) = 0;

    // Backward signature with general Optimizer
    virtual Matrix<T> Backward(const Matrix<T>& grad) = 0;

    // Weight initialization
    virtual void WeightInitialization(const WeightInit& technique) = 0;
};

#endif