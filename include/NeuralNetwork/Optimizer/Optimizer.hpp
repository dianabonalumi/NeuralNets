#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../Matrix.hpp"

#include "../../matrix_solver.hpp"

#include <memory>

template <typename T>
class Optimizer {
protected:
    const std::shared_ptr<Matrix_Solver<T>>& solver;

public:
    Optimizer(const std::shared_ptr<Matrix_Solver<T>>& solver): solver(solver) {}

    virtual void Optimize(Matrix<T>&, const Matrix<T>&) = 0;
};

#endif