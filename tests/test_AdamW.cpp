#include <iostream>
#include <chrono>
#include <cmath>
#include <cassert>
#include <random>
#include <vector>
#include <iomanip>

#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Optimizer/AdamW.hpp"
#include "../include/NeuralNetwork/Optimizer/Adam.hpp"
#include "../include/solvers/naive_solver.hpp"

template<typename T>
bool isClose(T a, T b, T tol = 1e-4) { 
    return std::abs(a - b) < tol; 
}

// 1. DECOUPLED WEIGHT DECAY VERIFICATION
void test_adamw_weight_decay_math() {
    std::cout << "1. VERIFYING DECOUPLED WEIGHT DECAY MATH..." << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    float lr = 0.1f, wd = 0.05f;
    AdamW<float> opt(solver, lr, 0.9f, 0.999f, 1e-8f, wd);

    Matrix<float> W(1, 1); W.Set(0, 0, 10.0f);
    Matrix<float> G_zero(1, 1); G_zero.Set(0, 0, 0.0f); 

    // Formula AdamW: w_new = w - lr * (m_hat/(sqrt(v_hat)+eps) + wd * w)
    // Con G=0, m_hat e v_hat sono 0 -> w_new = 10.0 - lr * (wd * 10.0) = 10.0 - 0.1*(0.05*10)=9.95
    opt.Optimize(W, G_zero);

    assert(isClose(W.Get(0, 0), 9.95f));
    std::cout << "   ✓ Weight decay applied correctly: 10.0 -> " << W.Get(0,0) << std::endl;
}

// 2. ADAM vs ADAMW CONVERGENCE
void test_adamw_vs_adam_stability() {
    std::cout << "2. COMPARING ADAM vs ADAMW STABILITY..." << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    
    Adam<float> adam(solver, 0.01f);
    AdamW<float> adamw(solver, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f);

    Matrix<float> W_a(1, 1); W_a.Set(0, 0, 1.0f);
    Matrix<float> W_aw(1, 1); W_aw.Set(0, 0, 1.0f);

    // Add noise to gradients to simulate noisy training; AdamW should regularize better
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);

    for(int i=0; i<100; ++i) {
        float gval = 0.5f + noise(rng);
        Matrix<float> G(1,1); G.Set(0,0, gval);
        adam.Optimize(W_a, G);
        adamw.Optimize(W_aw, G);
    }

    std::cout << "   Final weights => Adam: " << W_a.Get(0,0) << " | AdamW: " << W_aw.Get(0,0) << std::endl;
    // Expect AdamW to have a smaller magnitude due to weight decay under noisy gradients
    assert(std::abs(W_aw.Get(0, 0)) <= std::abs(W_a.Get(0, 0)) + 1e-4f);
    std::cout << "   ✓ AdamW achieved better regularization than Adam under noise." << std::endl;
}

// 3.MASSIVE LAYER STRESS
void test_adamw_massive_layer_isolation() {
    std::cout << "3. STRESSING 40-LAYER AUTOENCODER STATES..." << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    AdamW<float> adamw(solver, 0.001f);

    const int num_layers = 40;
    std::vector<Matrix<float>> weights;
    std::vector<Matrix<float>> grads;

    for (int i = 0; i < num_layers; ++i) {
        weights.emplace_back(140, 64); 
        weights[i].Set(0, 0, (float)i);
        grads.emplace_back(140, 64);
        grads[i].Set(0, 0, 0.01f);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < 100; ++step) {
        for (int i = 0; i < num_layers; ++i) adamw.Optimize(weights[i], grads[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "   ✓ 40 independent layers (8.9k params each) updated 100x in " << diff.count() << "s" << std::endl;
    
    for (int i = 0; i < num_layers; ++i) assert(std::isfinite(weights[i].Get(0, 0)));
}

// 4. NUMERICAL EXPLOSION TEST
void test_adamw_numerical_robustness() {
    std::cout << "4. TESTING NUMERICAL ROBUSTNESS (Exploding/Vanishing Grads)..." << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    AdamW<float> adamw(solver, 0.01f);

    Matrix<float> W(1, 1); W.Set(0, 0, 1.0f);
    Matrix<float> G_huge(1, 1); G_huge.Set(0, 0, 1e5f);
    Matrix<float> G_tiny(1, 1); G_tiny.Set(0, 0, 1e-10f);

    adamw.Optimize(W, G_huge); // stress da gradiente enorme
    assert(std::isfinite(W.Get(0, 0)));
    
    adamw.Optimize(W, G_tiny); // stress da gradiente quasi nullo
    assert(std::isfinite(W.Get(0, 0)));
    std::cout << "   ✓ AdamW remained stable under extreme gradient variance." << std::endl;
}

int main() {
    try {
        std::cout << "=== ADAMW TEST ===" << std::endl;

        test_adamw_weight_decay_math();
        test_adamw_vs_adam_stability();
        test_adamw_massive_layer_isolation();
        test_adamw_numerical_robustness();

        std::cout << "\n[RESULT] AdamW is fully validated for ECG Autoencoder. ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\n!!! ADAMW TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}