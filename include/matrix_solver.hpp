#ifndef MATRIX_SOLVER_HPP
#define MATRIX_SOLVER_HPP

#include <string>

template <typename T>
class Matrix_Solver {
public:
    virtual ~Matrix_Solver() = default;

    // Pure Virtual Method
    virtual void multiply(int M, int N, int K, const T* A, const T* B, T* C) = 0;
 
    virtual std::string getName() const = 0;
};

#endif
