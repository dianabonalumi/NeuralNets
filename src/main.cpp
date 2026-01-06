#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/DataLoader/DataLoader.hpp"
#include "../include/NeuralNetwork/Architecture/Autoencoder.hpp"
#include "../include/NeuralNetwork/Loss/MSE.hpp"

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

void train(std::shared_ptr<Architecture<float>>& arch, DataLoader<float>& data_train, DataLoader<float>& data_val, 
    std::shared_ptr<Loss<float>>& loss, int num_epochs) {

    std::ofstream loss_file("loss_log.csv");
    loss_file << "epoch,train_loss,eval_loss\n";

    float tr_loss, val_loss;

    int num_tr = data_train.totalSamples(), num_val = data_val.totalSamples();

    for(int i = 0;i < num_epochs;i++) {
        tr_loss = 0;

        data_train.shuffle();
        while(!data_train.isFinished()) {
            Matrix<float> X = data_train.getBatch().first;

            Matrix<float> out = arch->Predict(X);
            tr_loss += loss->Compute(out, X).Get(0,0);
            Matrix<float> grad = loss->Gradient();
            arch->Backward(X, grad);
        }

        val_loss = 0;

        data_val.shuffle();
        while(!data_val.isFinished()) {
            Matrix<float> X = data_val.getBatch().first;

            Matrix<float> out = arch->Predict(X);
            val_loss += loss->Compute(out, X).Get(0,0);
            Matrix<float> grad = loss->Gradient();
            arch->Backward(X, grad);
        }

        loss_file << i + 1 << "," << tr_loss / (float)num_tr << "," << val_loss / (float)num_val << std::endl;
    }
}

void test(std::shared_ptr<Architecture<float>>& arch, DataLoader<float>& data_test, 
    std::shared_ptr<Loss<float>>& loss) {

    float test_loss = 0;

    int n = data_test.totalSamples();

    data_test.shuffle();
    while(!data_test.isFinished()) {
        Matrix<float> X = data_test.getBatch().first;

        Matrix<float> out = arch->Predict(X);
        test_loss += loss->Compute(out, X).Get(0,0);
    }

    std::ofstream loss_file("test_loss_log.csv");
    loss_file << "test_loss\n";
    loss_file << test_loss / (float)n << std::endl;
}

int main() {
    std::shared_ptr<Matrix_Solver<double>> solver = std::move(SolverFactory<double>::createSolver(SolverType::ALL));

    DataLoader<float> data_train(1);
    data_train.loadCSV("dataset2/train.csv", "dataset2/train_labels.csv");

    DataLoader<float> data_val(1);
    data_val.loadCSV("dataset2/val.csv", "dataset2/val_labels.csv");

    std::shared_ptr<Architecture<float>> arch = std::make_shared<Autoencoder<float>>();
    std::shared_ptr<Loss<float>> loss = std::make_shared<MSE<float>>(solver);

    train(arch, data_train, data_val, loss, NUM_EPOCHS);

    DataLoader<float> data_test(1);
    data_test.loadCSV("dataset2/test.csv", "dataset2/test_labels.csv");
    test(arch, data_test, loss);
}