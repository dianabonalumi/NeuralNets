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
    
    virtual Matrix<T> Forward(const Matrix<T> X) = 0;
    virtual Matrix<T> Backward(const Matrix<T> grad) = 0;
};

#endif