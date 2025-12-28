# Neural Network Layer Module

This module serves as the computational core of the Neural Network architecture, defining the fundamental building blocks (Layers) and the mechanisms for signal propagation and learning. It implements the two core processes of a neural network: the **Forward Pass** (prediction) and the **Backward Pass** (gradient calculation and weight update).

The entire system is templated (`<typename T>`) to support various data types such as `float` or `double` and relies on external components for efficient matrix algebra.

---

## Module Structure and Components

The module consists of an abstract interface and several concrete implementations that handle linear transformations and non-linear activations:

* **`Layer.hpp`**: The abstract interface defining the contract for all computational stages.
* **`Dense.hpp`**: Implementation of the primary linear transformation layer (Fully Connected).
* **`Sigmoid.hpp`**: Implementation of the Sigmoid non-linear activation function.
* **`ReLU.hpp`**: Implementation of the Rectified Linear Unit (ReLU) non-linear activation function.
* **`Softmax.hpp`**: Implementation of the Softmax activation layer for probability distribution.
* **`WeightInitialization.hpp`**: Definitions for standard weight initialization techniques.

---

## 1. The Layer Abstraction (`Layer.hpp`)

`Layer.hpp` establishes the fundamental interface that every component in the neural network pipeline must adhere to. This abstraction ensures that different layer types can be chained together seamlessly.

### Core Contract
* **`Forward(const Matrix<T>& X)`**: Defines the data flow. It takes an input matrix $X$ (including the batch dimension) and returns the layer's output matrix.
* **`Backward(const Matrix<T>& grad)`**: Defines the error flow. The layer calculates the gradient with respect to its own input ($\frac{\partial L}{\partial X}$) and, if applicable, delegates weight updates to an internal **Optimizer**.
* **`WeightInitialization(const WeightInit& technique)`**: A mandatory method for all layers to handle weight setup according to modern standards (He or Xavier).

---

## 2. Weight Initialization (`WeightInitialization.hpp`)

Proper initialization is critical to prevent numerical instability, such as "Vanishing" or "Exploding" gradients, during the early stages of training. The module provides an enumeration to select the appropriate strategy based on the layer's activation function.


### Supported Techniques
* **Xavier (Glorot) Initialization**: Designed for layers with symmetric activation functions like **Sigmoid** or **Tanh**. It maintains a stable signal variance by scaling weights based on both input and output dimensions: $\sigma = \sqrt{\frac{2}{in + out}}$.
* **He Initialization**: Specifically optimized for **ReLU** activation functions. Since ReLU "shuts off" half of the input space, this technique compensates by using a larger variance based solely on the input dimension: $\sigma = \sqrt{\frac{2}{in}}$.

In the `Dense` layer implementation, these techniques utilize a normal distribution to break symmetry and ensure neurons learn distinct features.

---

## 3. The Linear Transformation Layer (`Dense.hpp`)

The `Dense` layer (Fully Connected) is responsible for the core weighted summation in the network.

### Mathematical Logic and Decoupled Optimization
**A. Forward Pass:**
Computes $Output = X \times W$. The input $X$ is cached as `lastInput` for use during the backward phase.

**B. Backward Pass (The Modular Approach):**
The architecture decouples **Gradient Calculation** from **Weight Updating**:
1.  **Gradient Propagation**: Calculates $\frac{\partial L}{\partial X}$ to inform previous layers.
2.  **Weight Gradient**: Calculates $\frac{\partial L}{\partial W}$ based on the cached `lastInput`.
3.  **Delegated Update**: Passes the calculated gradient to the assigned `Optimizer` (e.g., AdamW), allowing for advanced learning logic without bloating the layer code.

---

## 4. Activation Functions

Activation layers introduce non-linearity and operate **element-wise**. They do not possess learnable weights.

### A. Rectified Linear Unit (`ReLU.hpp`)
* **Forward Pass**: $f(x) = \max(0, x)$.
* **Backward Pass**: Passes the gradient through for positive inputs and blocks it (sets to zero) for non-positive inputs.

### B. Sigmoid Activation (`Sigmoid.hpp`)
* **Forward Pass**: $f(x) = \frac{1}{1 + e^{-x}}$.
* **Backward Pass**: The derivative is calculated using the cached output ($f(x)$): $grad \cdot (f(x) \cdot (1 - f(x)))$.

### C. Softmax Activation (`Softmax.hpp`)
Converts raw scores into a probability distribution.
* **Numerical Stability**: Implements a "Max Trick" (subtracting the row maximum before exponentiation) to prevent floating-point overflow.
* **Optimization Note**: Often paired with a Cross-Entropy loss function, simplifying the combined gradient to $(predictions - targets)$.