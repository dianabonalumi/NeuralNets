#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>
#include <random>

#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Optimizer/GradientDescent.hpp"
#include "../include/solvers/naive_solver.hpp"

template <typename T>
bool isClose(T a, T b, T tol = 1e-4) {
    return std::abs(a - b) < tol;
}

// 1. CONVERGENCE TEST
void stress_test_convergence() {
    std::cout << "1. Running Convergence Test" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    float lr = 0.05f;
    GradientDescent<float> opt(solver, lr);

    Matrix<float> W(1, 1);
    W.Set(0, 0, 0.0f); 

    float prev_loss = 1000.0f;
    for(int i = 0; i < 100; ++i) {
        float current_w = W.Get(0, 0);
        float loss = std::pow(current_w - 7.0f, 2);
        
        // calcolo del gradiente: dLoss/dW = 2 * (W - 7)
        Matrix<float> G(1, 1);
        G.Set(0, 0, 2.0f * (current_w - 7.0f));

        opt.Optimize(W, G); 

        if(i % 20 == 0) {
            std::cout << "Iteration " << i << ": W = " << W.Get(0,0) << " | Loss = " << loss << std::endl;
        }
        
        // la loss deve diminuire ad ogni step
        assert(loss <= prev_loss);
        prev_loss = loss;
    }

    assert(isClose(W.Get(0, 0), 7.0f, 0.1f));
    std::cout << "   ✓ Result: W converged to " << W.Get(0,0) << " (Target: 7.0)" << std::endl;
}

// 2. HIGH-DIMENSIONAL THROUGHPUT TEST
void stress_test_high_dim() {
    std::cout << "2. High-Dimensional Integrity (140x128)" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    float lr = 0.01f;
    GradientDescent<float> opt(solver, lr);

    int rows = 140, cols = 128; 
    Matrix<float> W(rows, cols);
    Matrix<float> G(rows, cols);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> d(-0.05f, 0.05f);

    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            W.Set(i, j, 1.0f);
            G.Set(i, j, d(rng));
        }
    }

    int si = 5, sj = 7;
    float g_sample = G.Get(si, sj);
    float expected_sample = W.Get(si, sj) - 10000 * lr * g_sample;

    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i = 0; i < 10000; ++i) {
        opt.Optimize(W, G); //
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "   ✓ Processed 10,000 updates of " << rows*cols << " elements in " << diff.count() << "s" << std::endl;
    
    assert(isClose(W.Get(si, sj), expected_sample, 1e-3f));
    for(int i = 0; i < rows; ++i)
        for(int j = 0; j < cols; ++j)
            assert(std::isfinite(W.Get(i,j)));
}

// 3. LEARNING RATE SENSITIVITY TEST
void stress_test_lr_stability() {
    std::cout << "3. Learning Rate Stability" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    
    float huge_lr = 10.0f;
    GradientDescent<float> opt_bad(solver, huge_lr);

    Matrix<float> W(1, 1); W.Set(0, 0, 1.0f);
    Matrix<float> G(1, 1); G.Set(0, 0, 1.0f);

    opt_bad.Optimize(W, G); 
    float val1 = W.Get(0, 0); // 1.0 - (10 * 1) = -9.0
    
    opt_bad.Optimize(W, G);
    float val2 = W.Get(0, 0); // -9.0 - (10 * 1) = -19.0

    assert(val2 < val1);
    std::cout << "   ✓ Divergence confirmed with high LR." << std::endl;
}
int main() {
    try {
        std::cout << "=== GRADIENT DESCENT TEST ===" << std::endl;

        stress_test_convergence();
        stress_test_high_dim();
        stress_test_lr_stability();
        std::cout << "\n[RESULT] All Gradient Descent tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n!!! GRADIENT DESCENT TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}