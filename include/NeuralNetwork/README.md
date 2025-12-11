# Neural Network Framework Documentation

This document provides a comprehensive technical overview of the custom C++ Neural Network framework. It details the project architecture, core data structures, computational logic, and the training execution flow.

The framework is designed with modularity and performance in mind, leveraging a custom Matrix library and an external optimized solver for heavy computational tasks.

## Project Structure Overview

The core logic is located within the `include/NeuralNetwork/` directory, organized into four specialized sub-modules:

* **`Matrix.hpp`**: The foundational data structure for numerical operations.
* **`Architecture/`**: Defines the network topology and execution flow (`Architecture.hpp`, `FeedForward.hpp`).
* **`Layer/`**: Implements the network building blocks (`Dense.hpp`, `ReLu.hpp`, `Sigmoid.hpp`).
* **`Loss/`**: Implements error evaluation metrics (`Loss.hpp`, `MSE.hpp`).
* **`DataLoader/`**: Handles efficient data input, shuffling, and batching (`DataLoader.hpp`).

Training execution is managed by `src/main.cpp`, while dataset preparation is handled by `dataset/Preprocessor.py`.

---

## I. Core Data Structures and Utilities

### 1. The Matrix Class (`Matrix.hpp`)
The `Matrix<T>` class is the backbone of the framework, designed to ensure memory safety and cache efficiency.

* **Memory Layout:** Data is stored in a contiguous 1D array using **Row-Major** order. This layout maximizes CPU cache locality and ensures compatibility with external `Matrix_Solver` implementations (e.g., SIMD/BLAS wrappers).
* **Resource Management:** The class implements the **Rule of Five** (Deep Copy Constructor, Copy Assignment, Move Constructor, Move Assignment, Destructor). This guarantees robust RAII (Resource Acquisition Is Initialization) memory management, preventing leaks during heavy training loops.
* **Key Operations:**
    * `Get(r, c)` / `Set(r, c)`: Maps 2D coordinates to the 1D internal buffer.
    * `Transpose()`: Essential for Backpropagation calculations.

### 2. Data Preprocessing (`dataset/Preprocessor.py`)
Before training, the raw dataset is processed via a Python script to ensure numerical stability.

* **Cleaning:** Removes samples containing NaN values and drops unnecessary categorical columns (e.g., `ocean_proximity`).
* **Splitting:** Randomly shuffles the data and splits it into Training (80%) and Testing (20%) sets.
* **Normalization:** Applies **Min-Max Scaling** to both features ($X$) and targets ($y$). Crucially, the scaling parameters are derived **only from the training set** to prevent data leakage.
* **Output:** Generates headerless CSV files (`X_train_scaled.csv`, etc.) optimized for parsing by the C++ `DataLoader`.

---

## II. Data Pipeline (`DataLoader.hpp`)

The `DataLoader` module provides the infrastructure for **iterative batch training** by managing data delivery.

* **Validation:** Ensures feature and label files are synchronized (same number of rows) upon loading.
* **Storage:** Parses CSV data into internal `Matrix<T>` objects for fast access.
* **Stochasticity (Shuffling):** Implements a `shuffle()` method that randomizes sample indices using a system-clock seeded engine. This is called before every epoch to ensure the model does not learn order-dependent patterns.
* **Batching:** The `getBatch()` method retrieves subsets of data (defined by `batch_size`), enabling iterative weight updates.

---

## III. Computational Model: Layers

The framework uses a polymorphic design where all layers inherit from the abstract `Layer` class. Each layer holds a pointer to a `Matrix_Solver` to delegate computationally intensive operations.

### 1. Dense Layer (`Dense.hpp`)
Implements a fully connected linear layer and handles parameter learning.

* **Forward Pass:** Computes the linear transformation $Y = X \times W$. It caches the input matrix (`lastInput`) for the backward pass.
* **Backward Pass:**
    1.  Computes the **Input Gradient** ($\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T$) to propagate error to the previous layer.
    2.  Computes the **Weight Gradient** ($\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}$).
    3.  **Update Rule:** Applies Gradient Descent immediately: $W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W}$.
* **Initialization:** Weights are initialized with small random values (uniform distribution between -0.1 and 0.1) to break symmetry.

### 2. Activation Functions
These layers introduce non-linearity element-wise.

* **ReLU (`ReLu.hpp`):**
    * *Forward:* $f(x) = \max(0, x)$.
    * *Backward:* Passes the gradient unchanged if $x > 0$, otherwise masks it to 0.
* **Sigmoid (`Sigmoid.hpp`):**
    * *Forward:* $f(x) = \frac{1}{1 + e^{-x}}$.
    * *Backward:* Uses the cached output to compute the derivative efficiently: $f'(x) = f(x)(1 - f(x))$.

---

## IV. Loss Functions (`Loss/`)

The module defines the objective function used to evaluate the network.

### Mean Squared Error (`MSE.hpp`)
Used for regression tasks.
* **Formula:** $L = 0.5 \sum (y_{pred} - y_{target})^2$. The factor $0.5$ is used to simplify the derivative.
* **Gradient:** The initial gradient propagated to the network is simply $(y_{pred} - y_{target})$.
* **Caching:** Stores predictions and targets during the forward pass to calculate gradients during the backward pass.

---

## V. Architecture and Execution

### 1. FeedForward Architecture (`FeedForward.hpp`)
Acts as the orchestrator for the training process.
* **Structure:** Contains a sequence of Layers (`std::vector<shared_ptr<Layer>>`) and a Loss function.
* **Training Cycle (`Train`):**
    1.  Executes `Forward` propagation through all layers.
    2.  Calculates the Loss and retrieves the initial gradient.
    3.  Executes `Backward` propagation in reverse order, updating weights in Dense layers.

### 2. Main Execution (`main.cpp`)
The entry point (`src/main.cpp`) instantiates the environment and runs the training loop.

* **Solver Configuration:** Instantiates a specific `Matrix_Solver` (e.g., `SIMD_UNROLL_2D`) via a factory pattern to optimize matrix operations.
* **Network Topology:** Defines a Deep Neural Network with **4 hidden layers** (Dense + ReLU) and a final output layer.
* **Training Loop:** Runs for a defined number of epochs (100). In each epoch:
    * Data is shuffled.
    * The model trains on batches using `ff.Train()`.
    * Performance is evaluated on the test set using `ff.Eval()`.
* **Logging:** Training and Validation losses are logged to `loss_log.csv` for analysis. Performance benchmarks (chronometers) are utilized to measure the efficiency of the matrix solvers.