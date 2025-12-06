#ifndef SOLVER_FACTORY_HPP
#define SOLVER_FACTORY_HPP

#include <memory>
#include "matrix_solver.hpp"
//solvers
#include "solvers/naive_solver.hpp"
#include "solvers/simd_solver.hpp"
#include "solvers/openmp_solver.hpp"
#include "solvers/loop_unroll_solver.hpp"
#include "solvers/tiling_solver.hpp"
#include "solvers/simd_unroll_solver_2D.hpp"
#include "solvers/simd_unroll_solver.hpp"
#include "solvers/tiling_openmp_solver.hpp"
#include "solvers/all_solver.hpp"


enum class SolverType {
    NAIVE,
    SIMD,
    UNROLL,
    TILING,
    OPENMP,
    SIMD_UNROLL_1D,
    SIMD_UNROLL_2D,
    TILING_OPENMP,
    ALL
};

template <typename T>
class SolverFactory {
public:
    static std::unique_ptr<Matrix_Solver<T>> createSolver(SolverType type = SolverType::NAIVE) {
        //switch over SolverType
        switch (type) {

            case SolverType::NAIVE:
                return std::make_unique<Naive_Solver<T>>();

            case SolverType::SIMD:
                return std::make_unique<Simd_Solver<T>>();

            case SolverType::OPENMP:
                    return std::make_unique<OpenMP_Solver<T>>();

            case SolverType::UNROLL:
                return std::make_unique<Loop_Unroll_Solver<T>>();

            case SolverType::TILING:
                return std::make_unique<Tiling_Solver<T>>();

            case SolverType::SIMD_UNROLL_1D:
                return std::make_unique<Simd_Unroll_Solver<T>>();

            case SolverType::SIMD_UNROLL_2D:
                return std::make_unique<Simd_Unroll_Solver_2D<T>>();

            case SolverType::TILING_OPENMP:
                return std::make_unique<Tiling_OpenMP_Solver<T>>();

            case SolverType::ALL:
                return std::make_unique<All_Solver<T>>();

            default:
                return std::make_unique<Naive_Solver<T>>();
        }
    }
};

#endif
