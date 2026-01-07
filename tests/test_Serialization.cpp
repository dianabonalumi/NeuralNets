#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/DataLoader/DataLoader.hpp"
#include "../include/NeuralNetwork/Architecture/Autoencoder.hpp"
#include "../include/NeuralNetwork/Loss/MSE.hpp"
#include "../include/NeuralNetwork/Optimizer/AdamW.hpp"

#include <iostream>

#include <memory>
#include <utility>

#include "../include/factory_m.hpp"

// g++ src/main.cpp -mavx -mfma -mavx2 -fopenmp -lpthread -o main

int main() {
    std::shared_ptr<Matrix_Solver<float>> solver = std::move(SolverFactory<float>::createSolver(SolverType::ALL));
    std::shared_ptr<Optimizer<float>> optim = std::make_shared<AdamW<float>>(solver);
    std::shared_ptr<Loss<float>> loss = std::make_shared<MSE<float>>(solver);

    std::shared_ptr<Autoencoder<float>> arch = std::make_shared<Autoencoder<float>>(
        solver, optim, loss, 140, 100, 128, 40, 20);

    arch->Save("models/test");
    std::cout << "Model saved" << std::endl;

    arch = std::make_shared<Autoencoder<float>>(
        solver, optim, loss, 140, 100, 128, 40, 20);

    arch->Restore("models/test");
    std::cout << "Model restored" << std::endl;
}