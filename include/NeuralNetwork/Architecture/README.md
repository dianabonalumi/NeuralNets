# Computational Architecture Module

This module defines the overarching structure of the Neural Network, establishing the core methods for training, evaluation, and prediction. It acts as the central orchestrator, linking the individual **Layer** components with the **Loss Function**.

---

### 1. Base Interface: `Architecture.hpp`

This is the abstract base class that sets the high-level contract for any type of network architecture (e.g., FeedForward, Recurrent, Convolutional).

* **Type:** Abstract Class (Template `<typename T>`).
* **Pure Virtual Methods (Model Contract):**
    * `virtual Matrix<T> Train(const Matrix<T>&, const Matrix<T>&, T) = 0;`: Executes a full forward and backward pass, updating parameters.
    * `virtual Matrix<T> Eval(const Matrix<T>&, const Matrix<T>&) = 0;`: Executes a forward pass and calculates the scalar loss value.
    * `virtual Matrix<T> Predict(const Matrix<T>&) = 0;`: Executes a forward pass to generate predictions.

---

### 2. Implementation: `FeedForward.hpp` (Sequential Architecture)

The `FeedForward` class implements a standard, sequential neural network architecture, typical for Multi-Layer Perceptrons (MLPs).

#### A. Structure and Initialization

The network structure is defined by two primary private members:

1.  `std::vector<std::shared_ptr<Layer<T>>> layers_`: A collection of all layers (Dense, Activation, etc.) that constitute the network's structure.
2.  `std::shared_ptr<Loss<T>> loss_`: The specific loss function (e.g., MSE) used to quantify error and calculate the initial gradient.

#### B. Core Methods Implementation

The `FeedForward` class orchestrates the training process:

| Method | Role | Implementation Flow |
| :--- | :--- | :--- |
| `Predict(X)` | **Forward Pass** | Iterates sequentially through the `layers_` vector, calling `Forward()` on each layer. The output of one layer becomes the input to the next, yielding the final network prediction. |
| `Eval(X, Y)` | **Loss Calculation** | 1. Calls `this->Predict(X)` to obtain network predictions (`pred`). 2. Calls `this->loss_->Compute(pred, Y)` to calculate the scalar error (loss value). |
| `Train(X, Y, learning_rate)` | **Backpropagation & Update** | 1. Executes the Forward Pass via `Eval(X, Y)`. 2. Initializes the gradient: `grad = this->loss_->Gradient()`. 3. **Iterates backwards** through the `layers_` vector (using reverse iterators `rbegin()` to `rend()`). 4. Calls `Backward(grad, learning_rate)` on each layer, updating the layer's parameters and propagating the gradient to the previous layer. |

The **`Train`** method is where the **optimization mechanism** is finalized, as it coordinates the entire cycle: Prediction, Loss Calculation, Initial Gradient computation, and the chain-rule application during the Backward Pass.