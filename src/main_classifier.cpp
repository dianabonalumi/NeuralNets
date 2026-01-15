#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/DataLoader/DataLoader.hpp"
#include "../include/NeuralNetwork/Architecture/Autoencoder.hpp"
#include "../include/NeuralNetwork/Loss/MSE.hpp"
#include "../include/NeuralNetwork/Optimizer/AdamW.hpp"

#include <iostream>

#include <memory>
#include <utility>

#include "../include/factory_m.hpp"

#define NUM_EPOCHS 50

// g++ src/main.cpp -mavx -mfma -mavx2 -fopenmp -lpthread -o main

void printMatrix(Matrix<float> m) {
    std::cout << "Matrix " << m.rows() << "x" << m.cols() << ":" << std::endl;
    for(int i = 0; i < m.rows(); i++) {
        for(int j = 0; j < m.cols(); j++) {
            std::cout << m.Get(i, j) << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "------------" << std::endl;
}

void test(std::shared_ptr<Autoencoder<float>>& arch, DataLoader<float>& data_test) {
    std::ofstream loss_file("classifier.csv");

    loss_file << "MSE,target,prediction\n";

    for(int i = 0;i < data_test.totalSamples();i++) {
        DataLoader<float>::Batch batch = data_test.getBatch();
        
        Matrix<float> X = batch.first, y = batch.second;

        Matrix<float> out = arch->Predict(X);
        out = arch->Eval(X);

        if(i % 10 == 0)
            std::cout << "Iteration " << i << "(class " << y.Get(0,0) << ") loss: " << out.Get(0,0) << "\n";

        loss_file << out.Get(0,0) << "," << y.Get(0,0) << "," << (out.Get(0,0) > 10.0) << "\n";
    }
}

int main() {
    std::shared_ptr<Matrix_Solver<float>> solver = std::move(SolverFactory<float>::createSolver(SolverType::ALL));
    std::shared_ptr<Optimizer<float>> optim = std::make_shared<AdamW<float>>(solver);
    std::shared_ptr<Loss<float>> loss = std::make_shared<MSE<float>>(solver);

    std::shared_ptr<Autoencoder<float>> arch = std::make_shared<Autoencoder<float>>(
        solver, optim, loss, 140, 30, 128, 40, 20);

    arch->Restore("models/best");
    std::cout << "Model restored" << std::endl;
    
    DataLoader<float> data_test(1);
    data_test.loadCSV("dataset2/test_classifier.csv", "dataset2/test_classifier_labels.csv");
    test(arch, data_test);
}