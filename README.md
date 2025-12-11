# NeuralNets

## Project Goal

The **main objective** of this project is to implement and benchmark highly **efficient and parallel Matrix Multiplication** routines from scratch using modern C++. The **Feed-Forward Neural Network (FFNN)** serves as the critical application platform where the performance and scalability of these custom solvers are rigorously tested during the training and inference processes.

The core design goal is to create a modular architecture that enables **scalable testing** of different matrix multiplication algorithms (Solvers). By decoupling the network structure from the computational backend, the project rigorously evaluates the performance gains, particularly the **speedup achieved through custom parallel matrix multiplication implementations** against standard single-threaded approaches.

## How to Run the Code

### Prerequisites

* C++ compiler
* OpenMP
* pthread
* AVX/AVX2/FMA
* Optional: python (Pandas) for preprocessing

### Build and Run Instructions

#### Optional (to perform our same testing procedure)
1. Download the dataset and move it to the *dataset* folder \
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