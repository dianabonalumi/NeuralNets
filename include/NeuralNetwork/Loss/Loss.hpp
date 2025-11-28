#ifndef LOSS_HPP
#define LOSS_HPP

#include "../Matrix.hpp"

#include "../../matrix_solver.hpp"

#include <memory>

template <typename T>
class Loss {
protected:
    std::shared_ptr<Matrix_Solver<T>> solver;

public:
    Loss(std::shared_ptr<Matrix_Solver<T>> solver): solver(solver) {}

    virtual Matrix<T> Compute(const Matrix<T>, const Matrix<T>) = 0;
    virtual Matrix<T> Gradient() = 0;

};

#endif