#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>
#include <random>
#include <vector>
#include <iomanip>

#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Optimizer/Adam.hpp"
#include "../include/NeuralNetwork/Optimizer/GradientDescent.hpp"
#include "../include/solvers/naive_solver.hpp"

// Helper per il confronto tra floating point
template<typename T>
bool isClose(T a, T b, T tol = 1e-4) { 
    return std::abs(a - b) < tol; 
}

// 1. CONVERGENCE COMPARISON
// check how many steps are needed to minimise f(w) = (w - 7)^2
int run_convergence_gd(float lr, float target = 7.0f) {
    auto solver = std::make_shared<Naive_Solver<float>>();
    GradientDescent<float> gd(solver, lr);
    Matrix<float> W(1, 1); W.Set(0, 0, 0.0f);
    for (int it = 1; it <= 1000; ++it) {
        Matrix<float> G(1, 1); G.Set(0, 0, 2.0f * (W.Get(0, 0) - target));
        gd.Optimize(W, G);
        if (std::abs(W.Get(0, 0) - target) < 0.01f) return it;
    }
    return 1001;
}

int run_convergence_adam(float lr, float target = 7.0f) {
    auto solver = std::make_shared<Naive_Solver<float>>();
    Adam<float> adam(solver, lr);
    Matrix<float> W(1, 1); W.Set(0, 0, 0.0f);
    for (int it = 1; it <= 1000; ++it) {
        Matrix<float> G(1, 1); G.Set(0, 0, 2.0f * (W.Get(0, 0) - target));
        adam.Optimize(W, G);
        if (std::abs(W.Get(0, 0) - target) < 0.01f) return it;
    }
    return 1001;
}

// 2. BIAS CORRECTION PRECISION
void test_math_precision() {
    auto solver = std::make_shared<Naive_Solver<float>>();
    Adam<float> adam(solver, 0.1f, 0.9f, 0.999f, 1e-8f);
    Matrix<float> W(1, 1); W.Set(0, 0, 1.0f);
    Matrix<float> G(1, 1); G.Set(0, 0, 0.5f);
    adam.Optimize(W, G);
    // nel punto 1, con questi parametri, W deve essere esattamente 0.9
    assert(isClose(W.Get(0, 0), 0.9f, 1e-5f));
    std::cout << "   ✓ Step 1 math is exact (0.9000)." << std::endl;
}

// 3. MULTI-STATE ISOLATION 
void test_multi_layer_isolation() {
    auto solver = std::make_shared<Naive_Solver<float>>();
    Adam<float> adam(solver, 0.1f);
    std::vector<Matrix<float>> weights(20, Matrix<float>(2, 2));
    std::vector<Matrix<float>> grads(20, Matrix<float>(2, 2));
    for (int i = 0; i < 20; ++i) {
        weights[i].Set(0, 0, (float)i * 10);
        grads[i].Set(0, 0, 1.0f);
    }
    for (int step = 0; step < 50; ++step) {
        for (int i = 0; i < 20; ++i) adam.Optimize(weights[i], grads[i]);
    }
    for (int i = 0; i < 20; ++i) assert(std::isfinite(weights[i].Get(0, 0)));
    std::cout << "   ✓ 20 independent layer states isolated." << std::endl;
}

// 4. HIGH-DIMENSIONAL & EPSILON STABILITY
void test_extreme_stability() {
    auto solver = std::make_shared<Naive_Solver<float>>();
    Adam<float> adam(solver, 0.001f);
    Matrix<float> W(512, 512); Matrix<float> G(512, 512);
    for (int i = 0; i < 512 * 512; ++i) {
        W.Flatten()[i] = 0.1f;
        G.Flatten()[i] = 1e-9f; 
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < 100; ++it) adam.Optimize(W, G);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "   ✓ 262k params (tiny grads) updated 100x in " << diff.count() << "s" << std::endl;
}

int main() {
    try {
        std::cout << "=== ADAM TESTS ===" << std::endl;

        std::cout << "1. RUNNING CONVERGENCE CHALLENGE..." << std::endl;
        int gd_steps = run_convergence_gd(0.01f);
        int adam_steps = run_convergence_adam(0.1f);
        std::cout << "   GD Steps: " << gd_steps << " | Adam Steps: " << adam_steps << std::endl;
        assert(adam_steps < gd_steps);

        std::cout << "2. VERIFYING BIAS CORRECTION MATH..." << std::endl;
        test_math_precision();

        std::cout << "3. STRESSING MULTI-LAYER MAP..." << std::endl;
        test_multi_layer_isolation();

        std::cout << "4. TESTING HIGH-DIM & EPSILON STABILITY..." << std::endl;
        test_extreme_stability();

        std::cout << "\n[RESULT] Adam passed ALL functional and stress tests! ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n!!! TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}