# Loss Functions Module

This module defines the **Loss Functions** used by the Neural Network to evaluate prediction errors and calculate the gradients necessary for learning.

The module relies on efficient memory management provided by the `Matrix` class.

## Module Structure

The contents of the `include/NeuralNetwork/Loss/` directory are as follows:

* **`Loss.hpp`**: Abstract Interface (Template Class).
* **`MSE.hpp`**: Mean Squared Error implementation.
* **`README.md`**: This documentation.



## Class Details
git 
### 1. Base Interface: `Loss.hpp`
This is the parent class from which all error metrics inherit.
* **Type:** Abstract Class (Template `<typename T>`).
* **Role:** Defines the contract ensuring interchangeability of loss functions without modifying the neural network code.
* **Pure Virtual Methods:**
    * `Compute()`: Calculates the scalar error (Forward pass).
    * `Gradient()`: Calculates the derivative of the error with respect to the output (Backward pass).

### 2. Implementation: `MSE.hpp` (Mean Squared Error)
This class implements the mathematical logic for regression problems.

#### Mathematical Logic
The class handles two fundamental steps:

**A. Forward Pass (Loss Calculation)**
Calculates the sum of squared errors. A scaling factor of 0.5 is applied to simplify the subsequent derivative.



**B. Backward Pass (Gradient Calculation)**
Calculates the partial derivative with respect to the prediction to initiate Backpropagation. Thanks to the 0.5 factor, the derivative becomes linear:



#### Memory Management
The `MSE` class internally stores the `lastX` (predictions) and `lastY` (target) matrices during the Forward pass, which are required to calculate the gradient in the Backward pass.

---

## External Dependencies: `Matrix.hpp`

Although `Matrix.hpp` is located in the parent directory (`include/Matrix.hpp`), it is the fundamental component upon which this module is based.

The `Loss` and `MSE` classes use `Matrix<T>` for all data operations. Key features of `Matrix.hpp` leveraged here include:

1.  **Resource Management (Rule of Five):**
    The `Matrix` class handles memory allocation and deallocation (RAII). This is crucial for `MSE`, which must save copies of matrices (`lastX`, `lastY`) without causing memory leaks or double-free errors.
    
2.  **Data Access:**
    Element access occurs via `Get(r, c)`, which maps 2D coordinates onto a contiguous 1D array to maximize CPU *cache locality* during error calculation loops.

