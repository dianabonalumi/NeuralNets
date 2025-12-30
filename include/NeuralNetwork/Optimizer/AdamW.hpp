#ifndef ADAMW_HPP
#define ADAMW_HPP

#include "Optimizer.hpp"
#include <unordered_map>
#include <map>
#include <cmath>

template <typename T>
class AdamW : public Optimizer<T> {
private:
    // Minimal AdamW optimizer (decoupled weight decay).
    T lr, beta1, beta2, epsilon, wd;
    int t;
    // Cache beta^t to avoid calling std::pow every update (more efficient & stable)
    T beta1_pow, beta2_pow;

    // Per-weight state (first and second moments), keyed by Matrix data pointer.
    std::unordered_map<T*, Matrix<T>> m;
    std::unordered_map<T*, Matrix<T>> v;

public:
        AdamW(std::shared_ptr<Matrix_Solver<T>> solver, T learning_rate = 0.001, 
                    T b1 = 0.9, T b2 = 0.999, T eps = 1e-8, T weight_decay = 0.01)
                : Optimizer<T>(solver), lr(learning_rate), beta1(b1), beta2(b2), 
                    epsilon(eps), wd(weight_decay), t(0), beta1_pow(static_cast<T>(1.0)), beta2_pow(static_cast<T>(1.0)) {}

        // Apply AdamW update in-place. weights and grad must match shapes.
    void Optimize(Matrix<T>& weights, const Matrix<T>& grad) override {
        // Basic sanity checks
        if (weights.rows() != grad.rows() || weights.cols() != grad.cols()) {
            throw std::runtime_error("AdamW: weights and grad must have the same shape");
        }

        T* w_ptr = weights.Flatten();
        const T* g_ptr = grad.Flatten();
        size_t size = weights.rows() * weights.cols();

        // Initialize state for this weight buffer on first use.
        // Matrix constructors in this codebase value-initialize data (zeros), so
        // newly created m and v matrices are zeroed by default.
        if (m.find(w_ptr) == m.end()) {
            m[w_ptr] = Matrix<T>(weights.rows(), weights.cols());
            v[w_ptr] = Matrix<T>(weights.rows(), weights.cols());
        }

        // Increment time-step and update cached powers
        t++;
        beta1_pow *= beta1;
        beta2_pow *= beta2;

        T* m_ptr = m[w_ptr].Flatten();
        T* v_ptr = v[w_ptr].Flatten();

        // Compute bias-correction denominators (1 - beta^t)
        T bias_corr1 = static_cast<T>(1.0) - beta1_pow;
        T bias_corr2 = static_cast<T>(1.0) - beta2_pow;

        for (size_t i = 0; i < size; ++i) {
            // Update running averages of gradient and squared gradient
            m_ptr[i] = beta1 * m_ptr[i] + (static_cast<T>(1.0) - beta1) * g_ptr[i];
            v_ptr[i] = beta2 * v_ptr[i] + (static_cast<T>(1.0) - beta2) * (g_ptr[i] * g_ptr[i]);

            // Bias-corrected estimates
            T m_hat = m_ptr[i] / bias_corr1;
            T v_hat = v_ptr[i] / bias_corr2;

            // Decoupled weight decay applied directly to parameters
            w_ptr[i] -= lr * (m_hat / (std::sqrt(v_hat) + epsilon) + wd * w_ptr[i]);
        }
    }
};

#endif