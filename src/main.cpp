#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Loss/MSE.hpp"
#include "../include/NeuralNetwork/Layer/Dense.hpp"
#include "../include/NeuralNetwork/Layer/ReLu.hpp"
#include "../include/NeuralNetwork/Architecture/FeedForward.hpp"
#include "../include/NeuralNetwork/DataLoader/DataLoader.hpp"

#include <iostream>

#include <memory>
#include <utility>

#include "../include/factory_m.hpp"

// To compile, from directory neuralnets-1-neuralnets/
// g++ src/main.cpp -mavx -mfma -mavx2 -o main

// Notes:
// For polymorphism pointers have to be used

void printMatrix(Matrix<double> m) {
    std::cout << "Matrix" << std::endl;
    for(int i = 0;i < m.rows();i++) {
        for(int j = 0;j < m.cols();j++) {
            std::cout << m.Get(i, j) << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "------------" << std::endl;
}

// M1: (a x b), M2: (b x c), res: (a x c), multiply: (a, c, b)

int main() {
    // solver
    std::shared_ptr<Matrix_Solver<double>> solver = std::move(SolverFactory<double>::createSolver(SolverType::SIMD_UNROLL_2D));

    // architecture shape
    int input_features = 8;

    int hidden_shape1 = 64;
    int hidden_shape2 = 32;
    int hidden_shape3 = 16;
    int hidden_shape4 = 8;

    int output_shape = 1;

    // data load
    DataLoader<double> train(1);
    train.loadCSV("/home/lorenzo/AMSC/neuralnets-1-neuralnets/dataset/X_train_scaled.csv",
        "/home/lorenzo/AMSC/neuralnets-1-neuralnets/dataset/y_train_scaled.csv");

    DataLoader<double> test(1);
    test.loadCSV("/home/lorenzo/AMSC/neuralnets-1-neuralnets/dataset/X_test_scaled.csv", 
        "/home/lorenzo/AMSC/neuralnets-1-neuralnets/dataset/y_test_scaled.csv");
     
    // layers
    std::shared_ptr<Dense<double>> input = std::make_shared<Dense<double>>(solver, input_features, hidden_shape1);
    std::shared_ptr<ReLU<double>> inp_act = std::make_shared<ReLU<double>>(solver);
    std::shared_ptr<Dense<double>> hidden1 = std::make_shared<Dense<double>>(solver, hidden_shape1, hidden_shape2);
    std::shared_ptr<ReLU<double>> active1 = std::make_shared<ReLU<double>>(solver);
    std::shared_ptr<Dense<double>> hidden2 = std::make_shared<Dense<double>>(solver, hidden_shape2, hidden_shape3);
    std::shared_ptr<ReLU<double>> active2 = std::make_shared<ReLU<double>>(solver);
    std::shared_ptr<Dense<double>> hidden3 = std::make_shared<Dense<double>>(solver, hidden_shape3, hidden_shape4);
    std::shared_ptr<ReLU<double>> active3 = std::make_shared<ReLU<double>>(solver);
    std::shared_ptr<Dense<double>> output = std::make_shared<Dense<double>>(solver, hidden_shape4, output_shape);

    std::vector<std::shared_ptr<Layer<double>>> layers({input, inp_act, hidden1, active1, hidden2, active2,
        hidden3, active3, output});

    // loss
    std::shared_ptr<Loss<double>> mse = std::make_shared<MSE<double>>(solver);

    // architecture
    FeedForward<double> ff(layers, mse);

    // logging losses
    std::ofstream loss_file("loss_log.csv");
    loss_file << "epoch,train_loss,eval_loss\n";

    for(int i = 0;i < 100;i++) {
        std::cout << "Starting epoch " << i + 1 << std::endl; 
        // training
        int n = train.totalSamples();
        double t_loss = 0;
        train.shuffle();
        while(!train.isFinished()) {
            std::pair<Matrix<double>, Matrix<double>> batch = train.getBatch();
            
            double res = ff.Train(batch.first, batch.second, 0.1).Get(0,0);

            t_loss += res / n;
        }

        // validation
        n = test.totalSamples();
        double val_loss = 0;
        test.shuffle();
        while(!test.isFinished()) {
            std::pair<Matrix<double>, Matrix<double>> batch = test.getBatch();

            double res = ff.Eval(batch.first, batch.second).Get(0,0);

            val_loss += res / n;
        }
        std::cout << "Train loss: " << t_loss << std::endl;
        std::cout << "Eval loss: " << val_loss << std::endl;

        loss_file << i+1 << "," << t_loss << "," << val_loss << "\n";
    }
    loss_file.close();
    return 0;
}