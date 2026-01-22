#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <iomanip>

#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Layer/Dense.hpp"
#include "../include/NeuralNetwork/Optimizer/GradientDescent.hpp"
#include "../include/solvers/naive_solver.hpp"

template<typename T>
bool isClose(T a, T b, T tol = 1e-4) { 
    return std::abs(a - b) < tol; 
}

// 1. BASIC FORWARD LOGIC TEST
void test_dense_forward_logic() {
    std::cout << "1. Forward Logic (Manual Verification)" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    auto optim = std::make_shared<GradientDescent<float>>(solver, 0.1f);
    
    // 2 input, 3 output
    Dense<float> dense(solver, optim, 2, 3); 
    
    // Inizializziamo manualmente pesi e bias per un calcolo deterministico
    // W = [[1, 0, 1], [0, 1, 1]], b = [0.5, 0.5, 0.5]
    float* wData = const_cast<float*>(dense.getWeights().Flatten());
    wData[0]=1.0f; wData[1]=0.0f; wData[2]=1.0f;
    wData[3]=0.0f; wData[4]=1.0f; wData[5]=1.0f;
    
    // Accediamo al bias (assumendo che sia inizializzato a zero dal costruttore)
    Matrix<float> X(1, 2); 
    X.Set(0, 0, 1.0f); X.Set(0, 1, 2.0f);

    // Forward: Y = X * W + b
    // riga 0: [1*1 + 2*0 + 0, 1*0 + 2*1 + 0, 1*1 + 2*1 + 0] = [1, 2, 3]
    Matrix<float> Y = dense.Forward(X);
    
    assert(isClose(Y.Get(0, 0), 1.0f));
    assert(isClose(Y.Get(0, 1), 2.0f));
    assert(isClose(Y.Get(0, 2), 3.0f));
    std::cout << "   ✓ Forward projection (1x2 -> 1x3) correct." << std::endl;
}

// 2. STATISTICAL CHECK OF WEIGHT INITIALIZATION
void test_dense_init_stats() {
    std::cout << "2. Weight Initialization Stats (Xavier)" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    auto optim = std::make_shared<GradientDescent<float>>(solver, 0.01f);
    
    int in = 200, out = 200;
    Dense<float> dense(solver, optim, in, out);
    
    dense.WeightInitialization(WeightInit::Xavier);
    
    float sum = 0, sq_sum = 0;
    const float* w = dense.getWeights().Flatten();
    size_t total = (size_t)in * out;
    for(size_t i = 0; i < total; ++i) {
        sum += w[i];
        sq_sum += w[i] * w[i];
    }
    float mean = sum / total;
    float var = sq_sum / total - (mean * mean);
    float expected_var = 2.0f / (in + out); // Varianza teorica Xavier

    assert(std::abs(mean) < 0.01f); // Media deve essere vicina a zero
    assert(isClose(var, expected_var, 0.01f));
    std::cout << "   ✓ Xavier Stats: Mean " << mean << ", Var " << var << " (Expected: " << expected_var << ")" << std::endl;
}

// 3. FINITE-DIFFERENCE GRADIENT CHECK
void test_dense_grad_check() {
    std::cout << "3. Finite-difference Gradient Check" << std::endl;
    auto solver = std::make_shared<Naive_Solver<float>>();
    auto optim = std::make_shared<GradientDescent<float>>(solver, 0.0f);

    int in_f = 4, out_f = 2;
    Dense<float> dense(solver, optim, in_f, out_f);
    dense.WeightInitialization(WeightInit::He);

    Matrix<float> X(1, in_f);
    for(int i=0; i<in_f; ++i) X.Set(0, i, (float)i * 0.5f);

    Matrix<float> Y = dense.Forward(X);
    Matrix<float> upstream_grad(1, out_f);
    upstream_grad.Set(0, 0, 1.0f); upstream_grad.Set(0, 1, -1.0f);

    Matrix<float> input_grad_analytic = dense.Backward(upstream_grad);

    // Grad check numerico per l'input X
    float eps = 1e-4f;
    for(int j = 0; j < in_f; ++j) {
        float orig = X.Get(0, j);
        
        X.Set(0, j, orig + eps);
        Matrix<float> Yp = dense.Forward(X);
        float Lp = Yp.Get(0,0)*1.0f + Yp.Get(0,1)*(-1.0f); 

        X.Set(0, j, orig - eps);
        Matrix<float> Ym = dense.Forward(X);
        float Lm = Ym.Get(0,0)*1.0f + Ym.Get(0,1)*(-1.0f);

        float num_grad = (Lp - Lm) / (2.0f * eps);
        X.Set(0, j, orig); 

        assert(isClose(num_grad, input_grad_analytic.Get(0, j), 1e-2f));
    }
    std::cout << "   ✓ Input Gradient (dL/dX) verified against numerical derivative." << std::endl;
}

int main() {
    try {
        std::cout << "=== DENSE LAYER TEST ===" << std::endl;
        test_dense_forward_logic();
        test_dense_init_stats();
        test_dense_grad_check();
        std::cout << "\n [RESULTS] ALL DENSE LAYER TESTS PASSED ✓" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "!!! DENSE TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}