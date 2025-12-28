#ifndef ADAM_HPP
#define ADAM_HPP

#include "Optimizer.hpp"
#include <unordered_map>
#include <map>
#include <cmath>

template <typename T>
class Adam : public Optimizer<T> {
private:
    T lr;          // Learning Rate
    T beta1;       // first moment decay
    T beta2;       // second moment decay
    T epsilon;     // small constant to avoid divide-by-zero
    int t;         // time step
    // Cache beta^t for bias correction
    T beta1_pow;
    T beta2_pow;

    // Per-parameter state keyed by the raw data pointer
    std::unordered_map<T*, Matrix<T>> m;
    std::unordered_map<T*, Matrix<T>> v;

public:
    Adam(std::shared_ptr<Matrix_Solver<T>> solver, T learning_rate = 0.001, 
         T b1 = 0.9, T b2 = 0.999, T eps = 1e-8)
        : Optimizer<T>(solver), lr(learning_rate), beta1(b1), beta2(b2), epsilon(eps), t(0), beta1_pow(static_cast<T>(1.0)), beta2_pow(static_cast<T>(1.0)) {}

    void Optimize(Matrix<T>& weights, const Matrix<T>& grad) override {
        T* w_ptr = weights.Flatten();
        const T* g_ptr = grad.Flatten();
        int size = static_cast<int>(weights.rows() * weights.cols());

        // Initialize state if first encounter
        if (m.find(w_ptr) == m.end()) {
            m[w_ptr] = Matrix<T>(weights.rows(), weights.cols());
            v[w_ptr] = Matrix<T>(weights.rows(), weights.cols());
        }

        // Increment time-step and cached powers
        t++;
        beta1_pow *= beta1;
        beta2_pow *= beta2;

        T* m_ptr = m[w_ptr].Flatten();
        T* v_ptr = v[w_ptr].Flatten();

        // Bias-correction denominators (1 - beta^t)
        T bias_corr1 = static_cast<T>(1.0) - beta1_pow;
        T bias_corr2 = static_cast<T>(1.0) - beta2_pow;

        for (int i = 0; i < size; ++i) {
            // Aggiornamento medie mobili
            m_ptr[i] = beta1 * m_ptr[i] + (1.0 - beta1) * g_ptr[i];
            v_ptr[i] = beta2 * v_ptr[i] + (1.0 - beta2) * (g_ptr[i] * g_ptr[i]);

            // Calcolo hat (correzione bias)
            T m_hat = m_ptr[i] / bias_corr1;
            T v_hat = v_ptr[i] / bias_corr2;

            // Aggiornamento finale dei pesi
            w_ptr[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
};

#endif