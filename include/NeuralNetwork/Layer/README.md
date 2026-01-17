# Neural Network Layer Module

This module serves as the computational core of the Neural Network architecture, defining the fundamental building blocks (Layers) and the mechanisms for signal propagation and learning. It implements the two core processes of a neural network: the **Forward Pass** (prediction) and the **Backward Pass** (gradient calculation and weight update).

The entire system is templated (`<typename T>`) to support various data types such as `float` or `double` and relies on external components for efficient matrix algebra.

---

## Module Structure and Components

The module consists of an abstract interface and several concrete implementations:

* **`Layer.hpp`**: The abstract interface defining the contract for all computational stages.
* **`Dense.hpp`**: Implementation of the primary linear transformation layer (Fully Connected).
* **`Sigmoid.hpp`, `ReLU.hpp`, `Softmax.hpp`**: Non-linear activation functions.
* **`CNN1D.hpp`**: 1D Convolutional layer for temporal or sequential feature extraction.
* **`MaxPooling1D.hpp`**: Downsampling layer for spatial variance reduction.
* **`LSTM.hpp`**: Long Short-Term Memory layer for handling long-range dependencies in sequences.
* **`WeightInitialization.hpp`**: Definitions for standard weight initialization techniques (He, Xavier).

---

## 1. The Layer Abstraction (`Layer.hpp`)

`Layer.hpp` establishes the fundamental interface. This ensures that different layer types (Linear, Convolutional, Recurrent) can be chained together seamlessly.

### Core Contract
* **`Forward(X)`**: Takes an input matrix $X$ and returns the layer's output.
* **`Backward(grad)`**: Calculates the gradient with respect to the input ($\frac{\partial L}{\partial X}$) and delegates parameter updates (Weights/Biases) to an internal **Optimizer**.
* **`WeightInitialization(technique)`**: Sets up weight variance based on the layer type and activation function to prevent vanishing/exploding gradients.

---

## 2. Linear and Activation Layers

### Dense Layer (`Dense.hpp`)
The primary linear transformation layer. It computes $Y = X \times W$. It decouples gradient calculation from weight updates by passing the weight gradient ($\frac{\partial L}{\partial W}$) to the assigned `Optimizer`.

### Activation Functions
* **ReLU**: $f(x) = \max(0, x)$. Provides sparse activation and prevents gradient saturation.
* **Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$. Maps values to a $(0, 1)$ range.
* **Softmax**: Converts raw scores into probabilities. Implements the **"Max Trick"** for numerical stability to prevent floating-point overflow during exponentiation.

---

## 3. Convolutional Components (1D)

Designed for sequential data where local patterns are shift-invariant.



### A. CNN1D (`CNN1D.hpp`)
* **Forward Pass**: Slides a set of learnable filters across the input sequence. Each filter performs a dot product to produce a feature map.
* **Backward Pass**: Computes gradients for the filters and the input data. Filter updates are managed by the optimizer to allow for adaptive learning.

### B. MaxPooling1D (`MaxPooling1D.hpp`)
* **Purpose**: Reduces the dimensionality of feature maps while retaining the most prominent features.
* **Logic**: Slides a window across the input and selects the maximum value.
* **Backward Pass**: A "Route" gradient approach—the gradient is passed back only to the index that produced the maximum value during the forward pass.

---

## 4. Recurrent Components: LSTM (`LSTM.hpp`)

The **Long Short-Term Memory** layer is designed to solve the vanishing gradient problem in standard RNNs by using a gated architecture.



### The Gating Mechanism
The LSTM maintains a **Cell State** ($c_t$) and a **Hidden State** ($h_t$) regulated by three gates:
1.  **Forget Gate**: Decides what information to discard from the cell state.
2.  **Input Gate**: Decides which new values to update in the cell state.
3.  **Output Gate**: Decides what the next hidden state should be based on the cell state.

### Implementation Details
* **Forward Pass**: Iterates through the time steps of the input sequence, updating the internal gates and states at each step.
* **Backward Pass (BPTT)**: Implements **Backpropagation Through Time**. Gradients are accumulated across all time steps to update the gate weights.
* **Complexity**: Managed via the `Matrix_Solver` to handle the multiple matrix multiplications required for the gate logic ($W_f, W_i, W_o, W_c$).

---

Would you like me to create a specific documentation section for the **WeightInitialization** logic applied to these new layers (e.g., how to initialize LSTM gates)?