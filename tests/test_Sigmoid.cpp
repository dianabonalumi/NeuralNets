#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Layer/Sigmoid.hpp"
#include "../include/solvers/naive_solver.hpp"

template<typename T>
bool isClose(T a, T b, T tol = 1e-5) { return std::abs(a - b) < tol; }

// 1. BASIC FORWARD AND BACKWARD LOGIC TEST
void test_sigmoid_logic() {
    std::cout << "1. Forward & Backward Logic" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    Sigmoid<float> sigmoid(solver);

    Matrix<float> X(1, 3);
    X.Set(0, 0, 0.0f);   // f(0) = 0.5
    X.Set(0, 1, 2.0f);   // f(2) approx 0.8807
    X.Set(0, 2, -2.0f);  // f(-2) approx 0.1192

    Matrix<float> out = sigmoid.Forward(X);
    
    assert(isClose(out.Get(0, 0), 0.5f));
    assert(out.Get(0, 1) > 0.8f);
    assert(out.Get(0, 2) < 0.2f);
    std::cout << "   ✓ Forward pass: S-curve mapping correct." << std::endl;

    // Test Backward: grad * (out * (1 - out))
    Matrix<float> grad(1, 3);
    grad.Set(0, 0, 1.0f); grad.Set(0, 1, 1.0f); grad.Set(0, 2, 1.0f);
    
    Matrix<float> inGrad = sigmoid.Backward(grad);
    
    // Per x=0, out=0.5, derivata = 1.0 * (0.5 * 0.5) = 0.25
    assert(isClose(inGrad.Get(0, 0), 0.25f));
    std::cout << "   ✓ Backward pass: Derivative calculation correct." << std::endl;
}

// 2. SATURATION STABILITY TEST
void test_sigmoid_saturation_stability() {
    std::cout << "2. Saturation Stability (Large Values)" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    Sigmoid<float> sigmoid(solver);

    Matrix<float> X(1, 2);
    X.Set(0, 0, 100.0f);  // Saturazione positiva -> 1.0
    X.Set(0, 1, -100.0f); // Saturazione negativa -> 0.0

    Matrix<float> out = sigmoid.Forward(X);
    
    assert(isClose(out.Get(0, 0), 1.0f));
    assert(isClose(out.Get(0, 1), 0.0f));
    
    // in saturazione la derivata deve essere quasi 0
    Matrix<float> grad(1, 2); grad.Set(0, 0, 1.0f); grad.Set(0, 1, 1.0f);
    Matrix<float> inGrad = sigmoid.Backward(grad);
    
    assert(inGrad.Get(0, 0) < 1e-5);
    assert(inGrad.Get(0, 1) < 1e-5);
    std::cout << "   ✓ Stability: No NaN/Inf on saturated inputs." << std::endl;
}

// Deterministic numeric regression check (high-precision reference)
// 3. NUMERIC REGRESSION CHECK
void test_sigmoid_numeric_regression() {
    std::cout << "3. Numeric Regression" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    Sigmoid<float> sigmoid(solver);

    Matrix<float> X(1, 5);
    const double vals[5] = {-2.0, -1e-6, 0.0, 1.0, 2.0};

    for (int i = 0; i < 5; ++i) X.Set(0, i, static_cast<float>(vals[i]));

    Matrix<float> out = sigmoid.Forward(X);

    for (int i = 0; i < 5; ++i) {
        double expected = 1.0 / (1.0 + std::exp(-vals[i]));
        assert(isClose(static_cast<float>(expected), out.Get(0, i), 1e-6f));
    }
    std::cout << "   ✓ Numeric regression against double precision reference passed." << std::endl;
}

// 4. FINITE-DIFFERENCE GRADIENT CHECK
void test_sigmoid_grad_check() {
    std::cout << "4. Finite-difference Gradient Check ---" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    Sigmoid<float> sigmoid(solver);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    const int R = 3, C = 4;
    Matrix<float> X(R, C);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            X.Set(i, j, dist(rng));

    // forward + backward with upstream = ones -> backprop should equal derivative
    Matrix<float> out = sigmoid.Forward(X);
    Matrix<float> upstream(R, C);
    for (int i = 0; i < R; ++i)
        for (int j = 0; j < C; ++j)
            upstream.Set(i, j, 1.0f);

    Matrix<float> back = sigmoid.Backward(upstream);

    const float eps = 1e-4f;
    const float tol = 1e-2f;

    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            float orig = X.Get(i, j);

            X.Set(i, j, orig + eps);
            Matrix<float> out_p = sigmoid.Forward(X);
            double Lp = 0.0;
            for (int ii = 0; ii < R; ++ii) for (int jj = 0; jj < C; ++jj) Lp += static_cast<double>(out_p.Get(ii, jj));

            X.Set(i, j, orig - eps);
            Matrix<float> out_m = sigmoid.Forward(X);
            double Lm = 0.0;
            for (int ii = 0; ii < R; ++ii) for (int jj = 0; jj < C; ++jj) Lm += static_cast<double>(out_m.Get(ii, jj));

            double num_grad = (Lp - Lm) / (2.0 * eps);

            X.Set(i, j, orig);
            sigmoid.Forward(X); 

            double back_grad = static_cast<double>(back.Get(i, j));
            double denom = std::max(1e-12, std::abs(num_grad) + std::abs(back_grad));
            double rel = std::abs(num_grad - back_grad) / denom;

            if (!(rel < tol)) {
                std::cerr << "Gradient check failed at (" << i << "," << j << ") : rel=" << rel << " num=" << num_grad << " back=" << back_grad << "\n";
            }
            assert(rel < tol);
        }
    }

    std::cout << "   ✓ Finite-difference gradient check passed (tol=" << tol << ")" << std::endl;
}
int main() {
    try {
        std::cout << "=== SIGMOID TEST ===" << std::endl;
        test_sigmoid_logic();
        test_sigmoid_saturation_stability();
        test_sigmoid_numeric_regression();
        test_sigmoid_grad_check();
        std::cout << "\n [RESULTS] ALL SIGMOID TESTS PASSED ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "!!! Sigmoid Test Failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}