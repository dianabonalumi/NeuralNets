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

void test(std::shared_ptr<Autoencoder<float>>& arch, DataLoader<float>& data_test, 
    std::shared_ptr<Loss<float>>& loss) {

    float test_loss = 0;

    int n = data_test.totalSamples();

    data_test.shuffle();
    int j = 0;
    while(!data_test.isFinished()) {
        Matrix<float> X = data_test.getBatch().first;

        Matrix<float> out = arch->Predict(X);
        test_loss += arch->Eval(X).Get(0,0);

        if(j % 10 == 0)
            std::cout << "Test loss: " << test_loss / (float)(j + 1) << "\n";
        j++;
    }

    std::ofstream loss_file("test_loss_log.csv");
    loss_file << "test_loss\n";
    loss_file << test_loss / (float)n << std::endl;
}

void plot_data(std::shared_ptr<Autoencoder<float>>& arch, DataLoader<float>& data_test) {
    data_test.shuffle();
    Matrix<float> X = data_test.getBatch().first;

    Matrix<float> out = arch->Predict(X);

    std::ofstream loss_file("plot.csv");
    loss_file << "predict,target\n";

    for(int i = 0;i < out.cols();i++) {
        loss_file << out.Get(0,i) << "," << X.Get(0,i) << "\n";
    }
}

int main() {
    std::shared_ptr<Matrix_Solver<float>> solver = std::move(SolverFactory<float>::createSolver(SolverType::ALL));
    std::shared_ptr<Optimizer<float>> optim = std::make_shared<AdamW<float>>(solver);
    std::shared_ptr<Loss<float>> loss = std::make_shared<MSE<float>>(solver);

    std::shared_ptr<Autoencoder<float>> arch = std::make_shared<Autoencoder<float>>(
        solver, optim, loss, 140, 100, 128, 40, 20);

    arch->Restore("models/best");
    std::cout << "Model restored" << std::endl;
    
    DataLoader<float> data_test(1);
    data_test.loadCSV("dataset2/test.csv", "dataset2/test_labels.csv");
    test(arch, data_test, loss);

    plot_data(arch, data_test);
}