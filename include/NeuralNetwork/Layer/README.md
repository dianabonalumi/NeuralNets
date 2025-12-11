# Neural Network Layer Module
This module serves as the computational core of the Neural Network architecture, defining the fundamental building blocks (Layers) and the mechanisms for signal propagation and learning. It implements the two core processes of a neural network: the **Forward Pass** (prediction) and the **Backward Pass** (gradient calculation and weight update).

The entire system is templated (`<typename T>`) to support various data types (`float`, `double`) and relies on external components for efficient matrix algebra.

---

## Module Structure and Components

The module is comprised of the following key files:

* **`Layer.hpp`**: The abstract interface defining the contract for all computational stages.
* **`Dense.hpp`**: Implementation of the primary linear transformation layer (Fully Connected).
* **`Sigmoid.hpp`**: Implementation of the Sigmoid non-linear activation function.
* **`ReLu.hpp`**: Implementation of the Rectified Linear Unit (ReLU) non-linear activation function.

---

## 1. The Layer Abstraction (`Layer.hpp`)

`Layer.hpp` establishes the fundamental interface that every component in the neural network pipeline must adhere to. This abstraction ensures that different layer types (linear, convolutional, activation) can be chained together seamlessly.

### Core Contract

* **`Forward(const Matrix<T> X)`**: Defines the data flow. Takes an input matrix $X$ (which includes the batch dimension) and returns the layer's output matrix.
* **`Backward(const Matrix<T> grad, T learning_rate)`**: Defines the error flow. Takes the incoming gradient from the subsequent layer and performs two main tasks:
    1.  Calculates the gradient with respect to its own input $\left( \frac{\partial L}{\partial X} \right)$, which is then passed backward.
    2.  If the layer has learnable parameters (like `Dense`), it updates them using the provided `learning_rate`.

### External Dependencies

All layers rely on a protected member, `std::shared_ptr<Matrix_Solver<T>> solver_`. This dependency allows the layer to delegate computationally intensive operations—specifically **matrix multiplication**—to an external, optimized backend (e.g., one leveraging BLAS or SIMD), thus separating high-level logic from low-level performance concerns.

---

## 2. The Linear Transformation Layer (`Dense.hpp`)

The `Dense` layer (also known as Fully Connected) is responsible for the core weighted summation in the network.

### Mathematical Logic

**A. Forward Pass:**
It performs the standard linear transformation on the input batch $X$ using the layer's internal weights $W$:
$\text{Output} = X \times W$
The input $X$ is cached internally as **`lastInput`** for gradient calculation.

**B. Backward Pass (Training):**
The `Dense` layer manages the full training cycle:

1.  **Input Gradient Calculation (Propagating to Previous Layer):**
    $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial \text{Output}} \times W^T$\
    This is achieved by multiplying the incoming gradient by the transpose of the weight matrix.

2.  **Weight Gradient Calculation (For Update):**
    $\frac{\partial L}{\partial W} = X^T \times \frac{\partial L}{\partial \text{Output}}$\
    This step requires the cached input $X$ (as $X^T$).

3.  **Weight Update (Gradient Descent):** The weights are updated using the calculated gradient and the provided learning rate:
    $W_{\text{new}} = W_{\text{old}} - \text{learning rate} \times \frac{\partial L}{\partial W}$\
    This is implemented element-wise.

### Weight Initialization

To prevent vanishing gradients and neuron saturation during early training, weights are initialized with small, random values uniformly distributed between $\mathbf{-0.1}$ and $\mathbf{0.1}$. This small-scale randomization is crucial to break symmetry and ensure different neurons learn distinct features.

---

## 3. Activation Functions

Activation layers introduce the necessary non-linearity, allowing the network to model complex, non-linear mappings. These layers operate **element-wise** and do not possess learnable weights.

### A. Rectified Linear Unit (`ReLu.hpp`)

The ReLU function is computationally efficient and has become the default activation for deep learning.

* **Forward Pass:** The output is the maximum of the input and zero:
    $f(x) = \max(0, x)$
* **Caching:** The original input matrix is stored as **`lastInput`**.
* **Backward Pass:** The gradient is passed through unmodified for positive inputs and blocked (set to zero) for non-positive inputs.
    $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial \text{Output}} \odot 1(x>0)$

### B. Sigmoid Activation (`Sigmoid.hpp`)

The Sigmoid function maps all values to a range between 0 and 1, making it historically popular for binary classification output layers.

* **Forward Pass:** Applies the logistic function:
    $f(x) = \frac{1}{1 + e^{-x}}$
* **Caching:** The layer stores its own output as **`lastOutput`**.
* **Backward Pass:** The derivative is elegantly calculated using the output itself, simplifying the process:
    $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial \text{Output}} \odot \left( \text{Output} \cdot (1 - \text{Output}) \right)$
