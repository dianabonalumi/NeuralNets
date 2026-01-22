# Optimizer Module

This module contains the optimizer implementations used to update model parameters during training. Optimizers encapsulate the parameter update logic (learning rate, momentum, adaptive steps, and weight decay) and operate directly on weight matrices via a shared interface.

---

## Module Structure

The module is built around a common interface and includes several concrete optimization strategies:

| File | Description |
| :--- | :--- |
| **`Optimizer.hpp`** | Abstract base class defining the core API: `Optimize(Matrix<T>& weights, const Matrix<T>& grad)`. |
| **`GradientDescent.hpp`** | Implements the standard Gradient Descent update rule (Vanilla GD). |
| **`Adam.hpp`** | Implementation of the Adaptive Moment Estimation algorithm. |
| **`AdamW.hpp`** | Adam variant with **Decoupled Weight Decay** for superior regularization. |

---

## Implemented Algorithms

### 1. Gradient Descent (Vanilla)
This is the most basic optimizer. It updates weights by moving them in the opposite direction of the gradient, scaled by a fixed learning rate ($lr$). It does not maintain any state (momentum or history).

**Update Rule:**
$$W_{t} = W_{t-1} - lr \cdot \nabla L$$

*Note: While often used in Stochastic Gradient Descent (SGD) contexts, this class specifically implements the mathematical update step.*

### 2. Adam (Adaptive Moment Estimation)
Adam maintains an exponential moving average of both the gradients ($m$) and the squared gradients ($v$) to compute individual adaptive learning rates for each parameter.

**Update Equations:**
1. **Momentum calculation:** $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$  
   $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
2. **Bias correction:** $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
3. **Weight update:** $W_t = W_{t-1} - lr \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

### 3. AdamW (Weight Decay)
AdamW modifies the standard Adam algorithm by decoupling the weight decay from the gradient update. This ensures that the regularization penalty is not distorted by the adaptive learning rates, leading to better generalization.

**Update Equation:**
$$W_t = W_{t-1} - lr \cdot \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda W_{t-1} \right)$$
*(Where $\lambda$ is the weight decay coefficient)*.

---

## Design Notes

- **State Management**: Adaptive optimizers store per-parameter state keyed by the raw `Flatten()` pointer of the `Matrix`. Current implementations use `std::unordered_map<T*, Matrix<T>>` (average O(1) lookup). This avoids adding state hooks to `Matrix` while assuming the internal pointer remains stable for the lifetime of the weights (no reallocation).
- **Numerical Stability & Performance**: Adaptive optimizers apply bias correction and add a small $\epsilon$ (default $10^{-8}$). To avoid expensive `std::pow` calls every step, implementations cache `\beta_1^t` and `\beta_2^t` via incremental multiplication for better performance and numerical stability.
- **Memory Efficiency**: The `Optimize` method mutates the `weights` matrix in-place, minimizing allocations during training.

**Note on GradientDescent**: `GradientDescent.hpp` provides a minimal, stateless update rule (vanilla GD). It is available as a simple baseline; unless explicitly attached to layers it will not be used automatically by higher-level training loops.

---

## Usage

Layers (such as `Dense`) are initialized with an `std::shared_ptr<Optimizer<T>>`. During the backward pass, the layer delegates the weight update to the optimizer:

```cpp
// Example: Initializing a Dense layer with AdamW
auto solver = std::make_shared<Simd_Solver<float>>();
auto opt = std::make_shared<AdamW<float>>(solver, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
auto layer = std::make_shared<Dense<float>>(solver, opt, 784, 128);