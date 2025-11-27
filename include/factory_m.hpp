#ifndef SOLVER_FACTORY_HPP
#define SOLVER_FACTORY_HPP

#include <memory>
#include "matrix_solver.hpp"

// Includi qui i vari solver separati
#include "solvers/naive_solver.hpp"
#include "solvers/simd_solver.hpp"
#include "solvers/openmp_solver.hpp"
#include "solvers/loop_unroll_solver.hpp"
#include "solvers/tiling_solver.hpp"


// #include "solvers/BlockedSolver.hpp" ...
enum class SolverType {
    NAIVE,
    SIMD,
    UNROLL,
    TILING,
    OPENMP
};

template <typename T>
class SolverFactory {
public:
    static std::unique_ptr<Matrix_Solver<T>> createSolver(SolverType type = SolverType::NAIVE) {
        switch (type) {

            case SolverType::NAIVE:
                return std::make_unique<Naive_Solver<T>>();

<<<<<<< HEAD
            case SolverType::SIMD:
                return std::make_unique<Simd_Solver<T>>();

            case SolverType::OPENMP:
                    return std::make_unique<OpenMP_Solver<T>>();
=======
            case SolverType::UNROLL:
                return std::make_unique<Loop_Unroll_Solver<T>>();

            case SolverType::TILING:
                return std::make_unique<Tiling_Solver<T>>();
>>>>>>> main
            
 

            default:
                return std::make_unique<Naive_Solver<T>>();
        }
    }
};

#endif
