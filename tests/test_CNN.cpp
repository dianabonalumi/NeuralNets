#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Loss/MSE.hpp"
#include "../include/NeuralNetwork/Layer/Dense.hpp"
#include "../include/NeuralNetwork/Layer/ReLu.hpp"
#include "../include/NeuralNetwork/Architecture/FeedForward.hpp"
#include "../include/NeuralNetwork/DataLoader/DataLoader.hpp"
#include "../include/NeuralNetwork/Layer/CNN1D.hpp"
#include "../include/NeuralNetwork/Layer/MaxPooling1D.hpp"
#include "../include/NeuralNetwork/Layer/Layer.hpp"
#include "../include/NeuralNetwork/Optimizer/GradientDescent.hpp"

#include <iostream>
#include <iomanip>

#include <memory>
#include <utility>

#include "../include/factory_m.hpp"

void printMatrix(Matrix<double> m) {
    std::cout << "Matrix " << m.rows() << "x" << m.cols() << ":" << std::endl;
    for(int i = 0; i < m.rows(); i++) {
        for(int j = 0; j < m.cols(); j++) {
            std::cout << m.Get(i, j) << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "------------" << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(4);
    
    // Test parameters
    int batch = 1;
    int in_channels = 2;
    int out_channels = 2;
    int kernel_size = 3;
    int length = 5;
    
    double data[10] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    Matrix<double> dm;
    dm.Unflatten(data, 1, 10);

    printMatrix(dm);

    auto solver = std::make_shared<Naive_Solver<double>>();
    auto optim = std::make_shared<GradientDescent<double>>(solver, 0.01f);

    auto cnn = CNN1D<double>(solver, optim, in_channels, out_channels, kernel_size, length);

    cnn.WeightInitialization(WeightInit::He);
    double* weights = cnn.getWeights().Flatten();

    for(int i = 0;i < kernel_size * out_channels * in_channels;i++) {
        weights[i] = 0;
    }
    weights[5] = 1;
    weights[6] = 1;

    printMatrix(cnn.getWeights());

    auto output = cnn.Forward(dm);
    printMatrix(output);

    // testing backward here
    Matrix<double> grad(batch, out_channels * length);
    double* gradData = grad.Flatten();
    // Different gradient pattern: [0.5, 0.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 0.5, 0.5]
    gradData[0] = 0.5; gradData[1] = 0.5;
    gradData[2] = 1.0; gradData[3] = 1.0;
    gradData[4] = 1.5; gradData[5] = 1.5;
    gradData[6] = 2.0; gradData[7] = 2.0;
    gradData[8] = 0.5; gradData[9] = 0.5;
    printMatrix(cnn.Backward(grad));

    // Test MaxPooling1D
    std::cout << "\n=== Testing MaxPooling1D ===" << std::endl;
    auto pool = MaxPooling1D<double>(solver, 2, 1, out_channels, length);
    auto pooled = pool.Forward(output);
    std::cout << "Pooled output:" << std::endl;
    printMatrix(pooled);

    // Test pooling backward
    Matrix<double> pool_grad(batch, pooled.cols());
    double* pool_grad_data = pool_grad.Flatten();
    std::fill(pool_grad_data, pool_grad_data + pool_grad.rows() * pool_grad.cols(), 1.0);
    auto pool_input_grad = pool.Backward(pool_grad);
    std::cout << "Pooling input gradient:" << std::endl;
    printMatrix(pool_input_grad);
}