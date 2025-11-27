#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Loss/MSE.hpp"
#include "../include/NeuralNetwork/Layer/Dense.hpp"
#include "../include/NeuralNetwork/Layer/Sigmoid.hpp"
#include "../include/NeuralNetwork/Architecture/FeedForward.hpp"

#include <iostream>

#include <memory>

// To compile, from directory neuralnets-1-neuralnets/
// g++ src/main.cpp src/NeuralNetwork/Loss/MSE.cpp src/NeuralNetwork/Layer/* src/NeuralNetwork/Architecture/FeedForward.cpp -o main

// Notes:
// For polymorphism pointers have to be used

int main() {
    Matrix<double> matrix(2);
    matrix[0] = 42;

    std::shared_ptr<Loss<double>> mse = std::make_unique<MSE<double>>();

    Matrix<double> test = mse->Compute(matrix);
    std::cout << test[0] << std::endl;

    std::shared_ptr<Layer<double>> dense = std::make_unique<Dense<double>>(1, 1);
    test = dense->Forward(matrix);

    std::cout << test[0] << std::endl;

    std::shared_ptr<Layer<double>> sigm = std::make_unique<Sigmoid<double>>();
    test = sigm->Forward(matrix);

    std::cout << test[0] << std::endl;

    std::vector<std::shared_ptr<Layer<double>>> layers({dense, sigm});

    std::unique_ptr<Architecture<double>> arch = std::make_unique<FeedForward<double>>(layers, mse);

    test = arch->Eval(matrix);

    std::cout << test[0] << std::endl;

    return 0;
}