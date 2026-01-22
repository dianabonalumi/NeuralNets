#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Layer/Softmax.hpp"
#include "../include/solvers/naive_solver.hpp"

template<typename T>
bool isClose(T a, T b, T tol = 1e-6) { return std::abs(a - b) < tol; }

// 1. BASIC FORWARD AND BACKWARD LOGIC TEST
void test_softmax_forward_basic() {
    std::cout << "1. Forward Basic Checks" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    Softmax<float> softmax(solver);

    Matrix<float> X(2, 3);
    X.Set(0,0, 0.0f); X.Set(0,1, 1.0f); X.Set(0,2, 2.0f);
    X.Set(1,0, 1000.0f); X.Set(1,1, 1001.0f); X.Set(1,2, 1002.0f);

    Matrix<float> out = softmax.Forward(X);

    for (int i = 0; i < out.rows(); ++i) {
        float s = 0.0f;
        for (int j = 0; j < out.cols(); ++j) {
            assert(out.Get(i,j) >= 0.0f);
            s += out.Get(i,j);
        }
        assert(isClose(s, 1.0f, 1e-6f));
    }

    // translation invariance: softmax(x + c) == softmax(x)
    Matrix<float> X2 = X;
    for (int i = 0; i < X2.rows(); ++i) for (int j = 0; j < X2.cols(); ++j) X2.Set(i,j, X2.Get(i,j) + 10.0f);
    Matrix<float> out2 = softmax.Forward(X2);
    for (int i = 0; i < out.rows(); ++i) for (int j = 0; j < out.cols(); ++j) assert(isClose(out.Get(i,j), out2.Get(i,j), 1e-6f));

    std::cout << "   ✓ Forward basic checks passed." << std::endl;
}

// 2. NUMERIC REGRESSION CHECK
void test_softmax_numeric_regression() {
    std::cout << "2. Numeric Regression" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    Softmax<float> softmax(solver);

    Matrix<float> X(1,4);
    const double vals[4] = {-2.0, -1.0, 0.0, 3.0};
    for (int j = 0; j < 4; ++j) X.Set(0,j, static_cast<float>(vals[j]));

    Matrix<float> out = softmax.Forward(X);

    double maxv = vals[0]; for (int j=1;j<4;++j) if (vals[j]>maxv) maxv = vals[j];
    double sum = 0.0; double tmp[4];
    for (int j=0;j<4;++j) { tmp[j] = std::exp(vals[j]-maxv); sum += tmp[j]; }
    for (int j=0;j<4;++j) {
        double expected = tmp[j] / sum;
        assert(isClose(static_cast<float>(expected), out.Get(0,j), 1e-6f));
    }

    std::cout << "   ✓ Numeric regression passed." << std::endl;
}

// 4. FINITE-DIFFERENCE GRADIENT CHECK
void test_softmax_grad_check() {
    std::cout << "4. Finite-difference Gradient Check" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    Softmax<float> softmax(solver);

    std::mt19937 rng(2023);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    const int R = 3, C = 5;
    Matrix<float> X(R, C);
    for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) X.Set(i, j, dist(rng));

    Matrix<float> out = softmax.Forward(X);
    Matrix<float> upstream(R, C);
    for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) upstream.Set(i, j, dist(rng));

    Matrix<float> back = softmax.Backward(upstream);

    const double eps = 1e-4;
    const double tol = 5e-1;

    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            double orig = X.Get(i,j);

            X.Set(i,j, orig + eps);
            Matrix<float> out_p = softmax.Forward(X);
            double Lp = 0.0; for (int ii=0;ii<R;++ii) for (int jj=0;jj<C;++jj) Lp += static_cast<double>(out_p.Get(ii,jj)) * static_cast<double>(upstream.Get(ii,jj));

            X.Set(i,j, orig - eps);
            Matrix<float> out_m = softmax.Forward(X);
            double Lm = 0.0; for (int ii=0;ii<R;++ii) for (int jj=0;jj<C;++jj) Lm += static_cast<double>(out_m.Get(ii,jj)) * static_cast<double>(upstream.Get(ii,jj));

            double num_grad = (Lp - Lm) / (2.0 * eps);

            X.Set(i,j, orig);
            softmax.Forward(X);

            double back_grad = static_cast<double>(back.Get(i,j));
            double denom = std::max(1e-12, std::abs(num_grad) + std::abs(back_grad));
            double rel = std::abs(num_grad - back_grad) / denom;

            const double rel_tol = tol; const double abs_tol = 1e-3;
            if (!((rel < rel_tol) || (std::abs(num_grad - back_grad) < abs_tol))) {
                std::cerr << "Softmax grad check failed at ("<<i<<","<<j<<") rel="<<rel<<" num="<<num_grad<<" back="<<back_grad<<"\n";
                std::cerr << "Row outputs: ";
                for (int jj=0; jj<C; ++jj) std::cerr << out.Get(i,jj) << ",";
                std::cerr << "\nUpstream: ";
                for (int jj=0; jj<C; ++jj) std::cerr << upstream.Get(i,jj) << ",";
                double dot = 0.0; for (int jj=0; jj<C; ++jj) dot += static_cast<double>(upstream.Get(i,jj)) * static_cast<double>(out.Get(i,jj));
                std::cerr << "\nDot = " << dot << "\n";
            }
            assert((rel < rel_tol) || (std::abs(num_grad - back_grad) < abs_tol));
        }
    }
    std::cout << "   ✓ Finite-difference gradient check passed (tol="<<tol<<")" << std::endl;
}

int main() {
    try {
        std::cout << "=== SOFTMAX TEST ===" << std::endl;
        test_softmax_forward_basic();
        test_softmax_numeric_regression();
        test_softmax_grad_check();
        std::cout << "\n [RESULTS] ALL SOFTMAX TESTS PASSED ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Softmax Test Failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
