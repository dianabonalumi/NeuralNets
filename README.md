# NeuralNets

## Project Goal

The **main objective** of this project is to implement and benchmark highly **efficient and parallel Matrix Multiplication** routines from scratch using modern C++.\
The **Feed-Forward Neural Network (FFNN)** serves as the critical application platform where the performance and scalability of these custom solvers are rigorously tested during the training and inference processes.

## How to Run the Code

### Prerequisites

* C++ compiler
* OpenMP
* pthread
* AVX/AVX2/FMA
* Optional: python (Pandas) for preprocessing

### Build and Run Instructions

#### Optional (to perform our same testing procedure)
1. Download the dataset, move it to the *dataset* folder and unzip it \
https://www.kaggle.com/datasets/camnugent/california-housing-prices

2. Run the preprocessor in the *dataset* folder to obtain processed data 
```bash
python dataset/Preprocessor.py
```

#### To run the program
1. Compile
```bash
g++ src/main.cpp -mavx -mfma -mavx2 -fopenmp -lpthread -o main
```

2. Run
```bash
./main
```

## File structure

The project directory structure is designed to separate the **neural network** from the **matrix solver** implementations.

**Note: Each folder contains its own `README.md` with a detailed explanation of its contents.**

```
NeuralNets
|-- dataset
|  |-- Preprocess.py
|  \-- dataset .csv files
|-- include
|  |-- NeuralNetwork
|  |  |-- Architecture
|  |  |  |-- Architecture.hpp
|  |  |  \-- FeedForward.hpp
|  |  |-- DataLoader
|  |  |  \-- DataLoader.hpp
|  |  |-- Layer
|  |  |  |-- Layer.hpp
|  |  |  |-- Dense.hpp
|  |  |  |-- ReLu.hpp
|  |  |  \-- Sigmoid.hpp
|  |  |-- Loss
|  |  |  |-- Loss.hpp
|  |  |  |-- MSE.hpp
|  |  \-- Matrix.hpp
|  |-- repo_mtx
|  |  \-- Report folders
|  |-- solvers
|  |  \-- Solver .hpp implementations
|  |-- utilities
|  |-- factory_m.hpp
|  \--matrix_solver.hpp
\-- src
   \-- main.cpp
```