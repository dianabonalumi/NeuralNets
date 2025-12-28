#ifndef ADAM_HPP
#define ADAM_HPP

#include "Optimizer.hpp"
#include <vector>
#include <map>
#include <cmath>

template <typename T>
class Adam : public Optimizer<T> {
private:
    T lr;          // Learning Rate
    T beta1;       // Decadimento primo momento (es. 0.9)
    T beta2;       // Decadimento secondo momento (es. 0.999)
    T epsilon;     // Per evitare divisioni per zero
    int t;         // Contatore dei passi (per la correzione del bias)

    // Adam deve ricordarsi i momenti m e v per ogni matrice di pesi che aggiorna.
    // Usiamo l'indirizzo della memoria dei dati come chiave univoca.
    std::map<T*, Matrix<T>> m; 
    std::map<T*, Matrix<T>> v;

public:
    Adam(std::shared_ptr<Matrix_Solver<T>> solver, T learning_rate = 0.001, 
         T b1 = 0.9, T b2 = 0.999, T eps = 1e-8)
        : Optimizer<T>(solver), lr(learning_rate), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

    void Optimize(Matrix<T>& weights, const Matrix<T>& grad) override {
        T* w_ptr = weights.Flatten();
        const T* g_ptr = grad.Flatten();
        int size = static_cast<int>(weights.rows() * weights.cols());

        // Se è la prima volta che questo ottimizzatore vede questi pesi, inizializza m e v
        if (m.find(w_ptr) == m.end()) {
            m[w_ptr] = Matrix<T>(weights.rows(), weights.cols());
            v[w_ptr] = Matrix<T>(weights.rows(), weights.cols());
            // newly constructed matrices are value-initialized to zero
        }

        t++; // Incrementiamo il tempo per la correzione del bias

        T* m_ptr = m[w_ptr].Flatten();
        T* v_ptr = v[w_ptr].Flatten();

        // Coefficienti per la correzione del bias
        T bias_corr1 = 1.0 - std::pow(beta1, t);
        T bias_corr2 = 1.0 - std::pow(beta2, t);

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