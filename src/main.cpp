#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Loss/MSE.hpp"
#include "../include/NeuralNetwork/Layer/Dense.hpp"
#include "../include/NeuralNetwork/Layer/Sigmoid.hpp"
#include "../include/NeuralNetwork/Architecture/FeedForward.hpp"

#include <iostream>

#include <memory>
#include <utility>

#include "../include/factory_m.hpp"

// To compile, from directory neuralnets-1-neuralnets/
// g++ src/main.cpp -mavx -mfma -mavx2 -o main

// Notes:
// For polymorphism pointers have to be used

int main() {
    // Initial owned matrix
    Matrix<double> m(2, 2);
    m.Set(0, 0, 1.0);

    // External data array (managed outside the Matrix class)
    double external_data[] = {10.0, 11.0, 12.0, 13.0, 14.1, 15.0}; 
    size_t external_rows = 2;
    size_t external_cols = 3;

    // Unflatten points 'm' to external_data (NO COPY)
    m.Unflatten(external_data, external_rows, external_cols);
    
    std::cout << "Value at (1, 1) before set: " << m.Get(1, 1) << std::endl; 
    
    // Modify external data via Matrix::Set (index 1*3 + 1 = 4)
    m.Set(1, 1, 99.9);
    
    std::cout << "Value at (1, 1) after set: " << m.Get(1, 1) << std::endl;
    
    // Verify that the external array was modified
    std::cout << "External data[4] is now: " << external_data[4] << std::endl; 

    std::shared_ptr solver = std::move(SolverFactory<double>::createSolver());  // needed to share the ptr

    std::shared_ptr<Loss<double>> mse = std::make_shared<MSE<double>>(solver);

    Matrix<double> test = mse->Compute(m);
    std::cout << test.Get(0,0) << std::endl;

    std::shared_ptr<Layer<double>> dense = std::make_shared<Dense<double>>(solver, 1, 1);
    test = dense->Forward(m);

    std::cout << test.Get(0,0) << std::endl;

    std::shared_ptr<Layer<double>> sigm = std::make_shared<Sigmoid<double>>(solver);
    test = sigm->Forward(m);

    std::cout << test.Get(0,0) << std::endl;

    std::vector<std::shared_ptr<Layer<double>>> layers({dense, sigm});

    std::unique_ptr<Architecture<double>> arch = std::make_unique<FeedForward<double>>(layers, mse);

    test = arch->Eval(m);

    std::cout << test.Get(0,0) << std::endl;

    return 0;
}