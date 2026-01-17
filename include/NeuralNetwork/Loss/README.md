# Loss Functions Module

This module defines the **Loss Functions** used by the Neural Network to evaluate prediction errors and calculate the gradients necessary for learning. It relies on efficient memory management provided by the `Matrix` class.

## Module Structure

The contents of the `include/NeuralNetwork/Loss/` directory are as follows:

* **`Loss.hpp`**: Abstract Interface (Template Class).
* **`MSE.hpp`**: Mean Squared Error implementation.
* **`CrossEntropy.hpp`**: Categorical Cross-Entropy implementation.
* **`README.md`**: This documentation.

---

## Class Details

### 1. Base Interface: `Loss.hpp`
This is the parent class from which all error metrics inherit.
* **Type:** Abstract Class (Template `<typename T>`).
* **Role:** Defines the contract ensuring interchangeability of loss functions without modifying the neural network code.
* **Pure Virtual Methods:**
    * `Compute()`: Calculates the scalar error (Forward pass).
    * `Gradient()`: Calculates the derivative of the error with respect to the output (Backward pass).

---

### 2. Implementation: `MSE.hpp` (Mean Squared Error)
Optimized for regression problems where the goal is to minimize the distance between continuous values.

#### Mathematical Logic
$$L = \frac{1}{2N} \sum (y_{pred} - y_{target})^2$$

* **Forward Pass:** Calculates the sum of squared errors. A scaling factor of $0.5$ is applied to simplify the derivative.
* **Backward Pass:** The derivative becomes linear: $\frac{\partial L}{\partial y_{pred}} = \frac{1}{N}(y_{pred} - y_{target})$.

---

### 3. Implementation: `CrossEntropy.hpp`
Typically used for classification tasks or probability distribution matching.

#### Mathematical Logic
$$L = -\frac{1}{N} \sum [y_{target} \cdot \log(y_{pred} + \epsilon)]$$

* **Numerical Stability:** Incorporates an $\epsilon$ (epsilon) constant ($10^{-9}$) to prevent $log(0)$ errors, which would result in undefined values.
* **Forward Pass:** Calculates the average logarithmic loss across the batch.
* **Backward Pass:** Computes the gradient: $\frac{\partial L}{\partial y_{pred}} = -\frac{1}{N} \cdot \frac{y_{target}}{y_{pred} + \epsilon}$.

#### Memory Management
Like the MSE implementation, this class stores `lastX` (predictions) and `lastY` (targets) to perform the gradient calculation efficiently without re-computing the forward pass.

---

## External Dependencies: `Matrix.hpp`

The `Loss` module leverages the following features of the `Matrix<T>` class:

1.  **Resource Management (Rule of Five):** The `Matrix` class handles RAII memory management, allowing `CrossEntropy` and `MSE` to store copies of input data safely.
2.  **Cache Locality:** By using `Flatten()`, the module accesses data as a contiguous 1D array, significantly improving performance by utilizing the CPU cache during large summation loops.
3.  **Exception Safety:** Includes checks to ensure that `prediction` and `target` matrices share identical dimensions before calculation.