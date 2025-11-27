#ifndef MSE_HPP
#define MSE_HPP

#include "Loss.hpp"

template <typename T>
class MSE : public Loss<T> {
private:
    Matrix<T> lastX;

public:
    Matrix<T> Compute(const Matrix<T>) override;
    Matrix<T> Gradient() override;
};

#endif