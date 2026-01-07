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

void train(std::shared_ptr<Autoencoder<float>>& arch, DataLoader<float>& data_train, DataLoader<float>& data_val, 
    std::shared_ptr<Loss<float>>& loss, int num_epochs) {

    std::ofstream loss_file("loss_log.csv");
    loss_file << "epoch,train_loss,eval_loss\n";

    float tr_loss, val_loss, best = 1000;

    int num_tr = data_train.totalSamples(), num_val = data_val.totalSamples();

    for(int i = 0;i < num_epochs;i++) {
        std::cout << "Epoch: " << i + 1 << std::endl;

        tr_loss = 0;

        data_train.shuffle();
        int j = 1;
        while(!data_train.isFinished()) {
            Matrix<float> X = data_train.getBatch().first;

            Matrix<float> out = arch->Predict(X);
            tr_loss += arch->Eval(X).Get(0,0);
            if(j % 10 == 1)
                std::cout << "Training iteration: " << j << "\tLoss: " << tr_loss / (float)j << std::endl; 
            arch->Backward();
            j++;
        }

        tr_loss /= (float)num_tr;
        std::cout << "Training loss: " << tr_loss << std::endl;

        val_loss = 0;

        j = 1;
        data_val.shuffle();
        while(!data_val.isFinished()) {
            Matrix<float> X = data_val.getBatch().first;

            Matrix<float> out = arch->Predict(X);
            val_loss += arch->Eval(X).Get(0,0);
            if(j % 10 == 1)
                std::cout << "Validation iteration: " << j << "\tLoss: " << val_loss / (float)j << std::endl; 
            arch->Backward();
            j++;
        }

        val_loss /= (float)num_val;
        std::cout << "Validation loss: " << val_loss << std::endl;

        loss_file << i + 1 << "," << tr_loss << "," << val_loss << std::endl;

        if(val_loss < best) {
            arch->Save("models/best");
            best = val_loss;
        }
    }
}

int main() {
    std::shared_ptr<Matrix_Solver<float>> solver = std::move(SolverFactory<float>::createSolver(SolverType::ALL));

    DataLoader<float> data_train(1);
    data_train.loadCSV("dataset2/train.csv", "dataset2/train_labels.csv");
    // data_train.loadCSV("../dataset2/train.csv", "../dataset2/train_labels.csv");

    DataLoader<float> data_val(1);
    data_val.loadCSV("dataset2/val.csv", "dataset2/val_labels.csv");
    // data_val.loadCSV("../dataset2/val.csv", "../dataset2/val_labels.csv");

    std::shared_ptr<Optimizer<float>> optim = std::make_shared<AdamW<float>>(solver, 0.0001);
    std::shared_ptr<Loss<float>> loss = std::make_shared<MSE<float>>(solver);

    std::shared_ptr<Autoencoder<float>> arch = std::make_shared<Autoencoder<float>>(
        solver, optim, loss, 140, 100, 128, 40, 20);

    train(arch, data_train, data_val, loss, NUM_EPOCHS);
    std::cout << "Train completed" << std::endl;
}