#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>

#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Layer/ReLu.hpp"
#include "../include/solvers/naive_solver.hpp"

template<typename T>
bool isClose(T a, T b, T tol = 1e-6) { return std::abs(a - b) < tol; }

// 1. BASIC FORWARD AND BACKWARD LOGIC TEST
void test_relu_forward_backward() {
    std::cout << "1. Forward & Backward Logic" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    ReLU<float> relu(solver);

    // 1. Test Forward: f(x) = max(0, x)
    Matrix<float> X(1, 4);
    X.Set(0, 0, -5.0f); X.Set(0, 1, 0.0f); X.Set(0, 2, 2.0f); X.Set(0, 3, -1.0f);

    Matrix<float> out = relu.Forward(X);
    
    assert(isClose(out.Get(0, 0), 0.0f));
    assert(isClose(out.Get(0, 1), 0.0f));
    assert(isClose(out.Get(0, 2), 2.0f));
    assert(isClose(out.Get(0, 3), 0.0f));
    std::cout << "   ✓ Forward pass: Negative values zeroed, positive kept." << std::endl;

    // 2. Test Backward: df/dx = 1 se x > 0, altrimenti 0
    Matrix<float> grad(1, 4);
    grad.Set(0, 0, 1.0f); grad.Set(0, 1, 1.0f); grad.Set(0, 2, 1.0f); grad.Set(0, 3, 1.0f);

    Matrix<float> inputGrad = relu.Backward(grad);

    // Il gradiente deve passare solo per l'indice 2 (dove X era 2.0)
    assert(isClose(inputGrad.Get(0, 0), 0.0f));
    assert(isClose(inputGrad.Get(0, 1), 0.0f));
    assert(isClose(inputGrad.Get(0, 2), 1.0f));
    assert(isClose(inputGrad.Get(0, 3), 0.0f));
    std::cout << "   ✓ Backward pass: Gradient masked correctly." << std::endl;
}

// 2. HIGH-DIMENSIONAL STRESS TEST
void test_relu_stress_high_dim() {
    std::cout << "2. High-Dimensional Stress (1024x1024)" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    ReLU<float> relu(solver);

    const int size = 1024;
    Matrix<float> X(size, size);
    float* data = X.Flatten();
    for(int i = 0; i < size * size; ++i) {
        data[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    Matrix<float> out = relu.Forward(X);
    
    Matrix<float> grad(size, size);
    std::fill(grad.Flatten(), grad.Flatten() + size*size, 1.0f);
    Matrix<float> inGrad = relu.Backward(grad);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "   ✓ Processed 1M elements (Forward+Backward) in " << diff.count() << "s" << std::endl;
    
    // Sanity check sulla somma
    float sum = 0;
    for(int i=0; i < size*size; ++i) sum += inGrad.Flatten()[i];
    assert(sum == (size * size / 2)); // Solo la metà positiva deve aver passato il gradiente
    std::cout << "   ✓ Total active neurons: " << sum << " (Correct: 50%)" << std::endl;
}

// 3. DETERMINISTIC NUMERIC REGRESSION CHECK
void test_relu_numeric_regression() {
    std::cout << "3. Numeric Regression (Deterministic)" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    ReLU<float> relu(solver);

    Matrix<float> X(1, 6);
    const float vals[6] = {-1.5f, -1e-6f, 0.0f, 0.5f, 2.0f, 10.0f};
    for (int i = 0; i < 6; ++i) X.Set(0, i, vals[i]);

    Matrix<float> out = relu.Forward(X);
    for (int i = 0; i < 6; ++i) {
        float expected = std::max(0.0f, vals[i]);
        assert(isClose(expected, out.Get(0, i), 1e-6f));
    }
    std::cout << "   ✓ Numeric regression passed." << std::endl;
}

// 4. FINITE-DIFFERENCE GRADIENT CHECK
void test_relu_grad_check() {
    std::cout << "4.Finite-difference Gradient Check" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    ReLU<float> relu(solver);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    const int R = 4, C = 5;
    Matrix<float> X(R, C);
    for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) {
        float v = dist(rng);
        // avoid near-zero to prevent non-diff point
        if (std::abs(v) < 1e-3f) v += (v < 0 ? -1e-2f : 1e-2f);
        X.Set(i, j, v);
    }

    Matrix<float> out = relu.Forward(X);
    Matrix<float> upstream(R, C);
    for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) upstream.Set(i, j, 1.0f);

    Matrix<float> back = relu.Backward(upstream);

    const float eps = 1e-4f;
    const float tol = 1e-3f;

    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            float orig = X.Get(i, j);
            X.Set(i, j, orig + eps);
            float Lp = 0.0f;
            Matrix<float> op = relu.Forward(X);
            for (int ii = 0; ii < R; ++ii) for (int jj = 0; jj < C; ++jj) Lp += op.Get(ii, jj);

            X.Set(i, j, orig - eps);
            Matrix<float> om = relu.Forward(X);
            float Lm = 0.0f;
            for (int ii = 0; ii < R; ++ii) for (int jj = 0; jj < C; ++jj) Lm += om.Get(ii, jj);

            float num_grad = (Lp - Lm) / (2.0f * eps);

            X.Set(i, j, orig);
            relu.Forward(X);

            float back_grad = back.Get(i, j);
            float denom = std::max(1e-8f, std::abs(num_grad) + std::abs(back_grad));
            float rel = std::abs(num_grad - back_grad) / denom;
            if (!(rel < tol)) std::cerr << "ReLU grad check failed at ("<<i<<","<<j<<") rel="<<rel<<" num="<<num_grad<<" back="<<back_grad<<"\n";
            assert(rel < tol);
        }
    }
    std::cout << "   ✓ Finite-difference gradient check passed." << std::endl;
}

int main() {
    try {
        std::cout << "=== RELU TEST ===" << std::endl;
        test_relu_forward_backward();
        test_relu_stress_high_dim();
        test_relu_numeric_regression();
        test_relu_grad_check();
        std::cout << "\n [RESULTS] ALL RELU TESTS PASSED ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "!!!! ReLU Test Failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}