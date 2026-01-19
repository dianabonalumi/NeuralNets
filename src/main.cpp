#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/DataLoader/DataLoader.hpp"
#include "../include/NeuralNetwork/Architecture/Autoencoder.hpp"
#include "../include/NeuralNetwork/Loss/MSE.hpp"
#include "../include/NeuralNetwork/Optimizer/AdamW.hpp"

#include <iostream>

#include <memory>
#include <utility>

#include "../include/factory_m.hpp"

#define NUM_EPOCHS 3
#define INPUT_SHAPE 140
#define BOTTLENECK 30
#define HIDDEN_SHAPE 128
#define WINDOW 40
#define STRIDE 20

#define LEARNING_RATE 0.0001

const double MAX_MSE = 15.0;

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

    std::ofstream loss_file("results/loss_log.csv");
    loss_file << "epoch,train_loss,val_loss\n";

    float tr_loss, val_loss, best = MAXFLOAT;

    int num_tr = data_train.totalSamples(), num_val = data_val.totalSamples();

    for(int i = 0;i < num_epochs;i++) {
        std::cout << "Epoch: " << i + 1 << std::endl;

        tr_loss = 0;

        data_train.shuffle();
        float j = 1;
        while(!data_train.isFinished()) {
            Matrix<float> X = data_train.getBatch().first;

            Matrix<float> out = arch->Predict(X);
            float val = arch->Eval(X).Get(0,0);

            tr_loss += val;

            if((int)j % 100 == 0)
                std::cout << "Training iteration: " << j << "\tLoss: " << tr_loss / j << std::endl; 
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
            if((int)j % 10 == 0)
                std::cout << "Validation iteration: " << j << "\tLoss: " << val_loss / j << std::endl; 
            j++;
        }

        val_loss /= (float)num_val;
        std::cout << "Validation loss: " << val_loss << std::endl;

        loss_file << i + 1 << "," << tr_loss << "," << val_loss << std::endl;
        loss_file.flush();

        // if(val_loss < best) {
        //     std::cout << "New best validation loss: " << val_loss << "\nModel saved!" << "\n";
        //     arch->Save("models/best");
        //     best = val_loss;
        // }
    }
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
    }

    test_loss /= (float)n;

    std::ofstream loss_file("results/test_loss_log.csv");
    loss_file << "test_loss\n";
    loss_file << test_loss << std::endl;

    std::cout << "Test loss: " << test_loss << std::endl;
}

void plot_data(std::shared_ptr<Autoencoder<float>>& arch, DataLoader<float>& data_test, int num, std::string name) {
    data_test.shuffle();
    for(int i = 0;i < num;i++) {
        DataLoader<float>::Batch b = data_test.getBatch();
        Matrix<float> X = b.first;
        int label = (int)b.second.Get(0,0);

        Matrix<float> out = arch->Predict(X);

        std::string f_name = "results/" + name;
        f_name += std::to_string(label);
        f_name += "_" + std::to_string(i);
        f_name += ".csv";

        std::ofstream loss_file(f_name);
        loss_file << "predict,target\n";

        for(int i = 0;i < out.cols();i++) {
            loss_file << out.Get(0,i) << "," << X.Get(0,i) << "\n";
        }
    }
}

void classify(std::shared_ptr<Autoencoder<float>>& arch, DataLoader<float>& data_test) {
    std::ofstream loss_file("results/classifier.csv");

    loss_file << "MSE,target,prediction\n";

    int tp = 0, tn = 0, fp = 0, fn = 0;

    data_test.shuffle();
    for(int i = 0;i < data_test.totalSamples();i++) {
        DataLoader<float>::Batch batch = data_test.getBatch();
        
        Matrix<float> X = batch.first;
        int y = (int)batch.second.Get(0,0);

        Matrix<float> out = arch->Predict(X);
        out = arch->Eval(X);

        float mse = out.Get(0,0);

        if(i % 10 == 0)
            std::cout << "Iteration " << i << "(class " << y << ") loss: " << mse << "\n";

        bool pred = mse < MAX_MSE;
        bool tr = y == 1;

        loss_file << out.Get(0,0) << "," << y << "," << pred << "\n";

        if(pred && tr)
            tp += 1;
        else if(pred && !tr)
            fp += 1;
        else if(!pred && tr)
            fn += 1;
        else
            tn += 1;
    }

    float prec = (float)tp / (float)(tp + fp);
    float rec = (float)tp / (float)(tp + fn);

    std::cout << "F1 Score: " << 2 * prec * rec / (prec + rec) << std::endl;
}

int main() {
    // Set global seed for reproducibility
    unsigned global_seed = 42;
    srand(global_seed);
    std::srand(global_seed);

    std::shared_ptr<Matrix_Solver<float>> solver = std::move(SolverFactory<float>::createSolver(SolverType::NAIVE));

    DataLoader<float> data_train(1);
    data_train.loadCSV("dataset2/train.csv", "dataset2/train_labels.csv");

    DataLoader<float> data_val(1);
    data_val.loadCSV("dataset2/val.csv", "dataset2/val_labels.csv");

    DataLoader<float> data_test(1);
    data_test.loadCSV("dataset2/test.csv", "dataset2/test_labels.csv");

    DataLoader<float> data_classifier(1);
    data_classifier.loadCSV("dataset2/test_classifier.csv", "dataset2/test_classifier_labels.csv");

    std::shared_ptr<Optimizer<float>> optim = std::make_shared<AdamW<float>>(solver, LEARNING_RATE);
    std::shared_ptr<Loss<float>> loss = std::make_shared<MSE<float>>(solver);

    std::shared_ptr<Autoencoder<float>> arch = std::make_shared<Autoencoder<float>>(
        solver, optim, loss, INPUT_SHAPE, BOTTLENECK, HIDDEN_SHAPE, WINDOW, STRIDE);

    train(arch, data_train, data_val, loss, NUM_EPOCHS);
    std::cout << "Train completed" << std::endl;

    std::cout << "Testing model" << std::endl;
    test(arch, data_test, loss);

    std::cout << "Generating normal reconstruction plots" << std::endl;
    plot_data(arch, data_test, 4, "normal");

    std::cout << "Classification" << std::endl;
    classify(arch, data_classifier);

    std::cout << "Generating classification reconstruction plots" << std::endl;
    plot_data(arch, data_classifier, 4, "classification");
}