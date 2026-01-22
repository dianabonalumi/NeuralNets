# Neural Network Framework Documentation

This document provides a comprehensive technical overview of the custom C++ Neural Network framework. It details the project architecture, core data structures, and the computational logic governing the system.

The framework is designed with modularity and performance in mind, leveraging a custom Matrix library and an external optimized solver for heavy computational tasks.

---

## I. Project Structure Overview

The core logic is located within the `include/NeuralNetwork/` directory, organized into specialized sub-modules:

* **`Matrix.hpp`**: The foundational data structure for numerical operations.
* **`Architecture/`**: Defines the network topology and execution flow (`Architecture.hpp`, `FeedForward.hpp`, `Autoencoder.hpp`).
* **`Layer/`**: Implements the network building blocks including Linear, Convolutional, and Recurrent types.
* **`Optimizer/`**: Encapsulates weight update logic (`Adam.hpp`, `AdamW.hpp`, `GradientDescent.hpp`).
* **`Loss/`**: Implements error evaluation metrics (`Loss.hpp`, `MSE.hpp`, `CrossEntropy.hpp`).
* **`DataLoader/`**: Handles efficient data input, shuffling, and batching.

---

## II. Core Data Structures and Utilities

### 1. The Matrix Class (`Matrix.hpp`)
The `Matrix<T>` class is the backbone of the framework, designed to ensure memory safety and cache efficiency.

* **Memory Layout:** Data is stored in a contiguous 1D array using **Row-Major** order. This maximizes CPU cache locality and ensures compatibility with external `Matrix_Solver` implementations.
* **Resource Management:** Implements the **Rule of Five** (Deep Copy, Move semantics, etc.) to guarantee robust RAII memory management.

### 2. Data Preprocessing (`dataset/Preprocessor.py`)
A Python pipeline handles numerical stability through **Min-Max Scaling** and data cleaning (NaN removal) before the C++ engine ingests the CSV files.

---

## III. Computational Model: Layers

The framework uses a polymorphic design where all layers inherit from an abstract `Layer<T>` class.

### 1. Linear and Basic Layers
* **Dense Layer (`Dense.hpp`):** A fully connected layer computing $Y = X \times W$. It decouples gradient calculation from updating logic by delegating weight adjustments to an internal `Optimizer`.
* **Activation Functions:** * **ReLU:** $f(x) = \max(0, x)$.
    * **Sigmoid:** $f(x) = \frac{1}{1 + e^{-x}}$.
    * **Softmax:** Maps raw scores to probabilities using the "Max Trick" for numerical stability.

### 2. Convolutional Layers (1D)
Designed for sequential or temporal data processing.
* **CNN1D (`CNN1D.hpp`):** Slides learnable filters across the input sequence to extract local features.
* **MaxPooling1D (`MaxPooling1D.hpp`):** Reduces spatial dimensions by selecting the maximum value within a sliding window. It uses a "route" gradient backpropagation (passing error only to the winning index).


### 3. Recurrent Layers
* **LSTM (`LSTM.hpp`):** A Long Short-Term Memory layer designed to capture long-range dependencies. It utilizes a gated architecture (Forget, Input, and Output gates) to manage a persistent cell state. It implements **Backpropagation Through Time (BPTT)** to accumulate gradients across the sequence.


---

## IV. Optimizer Module (`Optimizer/`)

Optimizers encapsulate the parameter update logic, allowing the user to swap strategies without changing layer code.

| Optimizer | Update Strategy | Key Features |
| :--- | :--- | :--- |
| **Gradient Descent** | $W_{t} = W_{t-1} - lr \cdot \nabla L$ | Minimal, stateless vanilla update. |
| **Adam** | Adaptive Moment Estimation | Uses moving averages of gradients ($m$) and squared gradients ($v$). |
| **AdamW** | Decoupled Weight Decay | Improves regularization by separating weight decay from the gradient update. |

---

## V. Loss Functions (`Loss/`)

* **Mean Squared Error (`MSE.hpp`):** Calculates $0.5 \sum (y_{pred} - y_{target})^2$.
* **Cross Entropy (`CrossEntropy.hpp`):** Used for classification; utilizes an $\epsilon$ ($1e-9$) buffer to prevent $log(0)$ errors.

---

## VI. Architectures

* **FeedForward (`FeedForward.hpp`):** A sequential container for linear data flow through layers.
* **Autoencoder (`Autoencoder.hpp`):** A specialized architecture split into an **Encoder** and a **Decoder**. It features custom binary serialization to save and restore the model weights into separate `.bin` files.