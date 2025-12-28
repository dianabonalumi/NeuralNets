#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <memory>
#include "../../Matrix.hpp"
#include "../../matrix_solver.hpp"

template <typename T>
class Optimizer {
protected:
    std::shared_ptr<Matrix_Solver<T>> solver;
public:
    Optimizer(std::shared_ptr<Matrix_Solver<T>> s) : solver(s) {}
    virtual ~Optimizer() = default;
    
    // Standard name used across implementations and layers
    virtual void Optimize(Matrix<T>& weights, const Matrix<T>& grad) = 0;
};

#endif