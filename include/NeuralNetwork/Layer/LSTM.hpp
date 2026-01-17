#ifndef LSTM_HPP
#define LSTM_HPP

#include "Layer.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <memory>
#include <algorithm>

template <typename T>
class LSTM : public Layer<T> {
private:
    int input_size, hidden_size;

    // Parameters: Packed [f, i, c_bar, o]
    Matrix<T> W_all, U_all, b_all;

    // Recurrent States
    Matrix<T> h_state, c_state;

    // BPTT History Stack
    struct StepCache {
        Matrix<T> x, prev_h, prev_c, gates, tanh_c;
    };
    std::vector<StepCache> history;

    // Numerical Stability Constants
    const T CLAMP_LIMIT = static_cast<T>(15.0); 
    const T STATE_LIMIT = static_cast<T>(50.0);

    // Helpers
    inline T sigmoid(T x) { 
        x = std::max(-CLAMP_LIMIT, std::min(CLAMP_LIMIT, x));
        return 1.0 / (1.0 + std::exp(-x)); 
    }
    inline T d_sigmoid(T y) { return y * (1.0 - y); }
    inline T safe_dtanh(T y) { 
        T val = 1.0 - (y * y);
        return std::max(static_cast<T>(0.0), std::min(static_cast<T>(1.0), val));
    }

    // Global Norm Gradient Clipping
    void clipGradients(Matrix<T>& dW, Matrix<T>& dU, Matrix<T>& db, T threshold) {
        T sum_sq = 0;
        auto check_and_sum = [&](Matrix<T>& m) {
            T* data = m.Flatten();
            size_t len = m.rows() * m.cols();
            for (size_t i = 0; i < len; ++i) {
                if (std::isnan(data[i]) || std::isinf(data[i])) data[i] = 0; // Heal NaNs
                sum_sq += data[i] * data[i];
            }
        };
        check_and_sum(dW); check_and_sum(dU); check_and_sum(db);
        T norm = std::sqrt(sum_sq);
        if (norm > threshold) {
            T scale = threshold / (norm + static_cast<T>(1e-7));
            auto apply = [&](Matrix<T>& m) {
                T* d = m.Flatten();
                for (size_t i = 0; i < m.rows() * m.cols(); ++i) d[i] *= scale;
            };
            apply(dW); apply(dU); apply(db);
        }
    }

public:
    LSTM(std::shared_ptr<Matrix_Solver<T>> solver, std::shared_ptr<Optimizer<T>> opt, int in_f, int hid_s)
        : Layer<T>(solver, opt), input_size(in_f), hidden_size(hid_s) {
        W_all = Matrix<T>(input_size, 4 * hidden_size);
        U_all = Matrix<T>(hidden_size, 4 * hidden_size);
        b_all = Matrix<T>(1, 4 * hidden_size);
        resetState();
    }

    void resetState() {
        h_state = Matrix<T>(0, 0);
        c_state = Matrix<T>(0, 0);
        history.clear();
    }

    void WeightInitialization(const WeightInit& technique) override {
        // Use fixed seed to ensure reproducibility
        std::default_random_engine gen(42);
        T scale = (technique == WeightInit::Xavier) ? 
            std::sqrt(2.0 / (input_size + hidden_size)) : std::sqrt(1.0 / input_size);
        
        std::normal_distribution<T> dist(0.0, scale);
        auto init = [&](Matrix<T>& m) {
            T* data = m.Flatten();
            for(size_t i = 0; i < m.rows() * m.cols(); ++i) data[i] = dist(gen);
        };
        init(W_all); init(U_all);
        
        T* b_ptr = b_all.Flatten();
        std::fill(b_ptr, b_ptr + (1 * 4 * hidden_size), 0);
        for(int j = 0; j < hidden_size; ++j) b_ptr[j] = 1.0; // Forget gate bias
    }

    Matrix<T> Forward(const Matrix<T>& X) override {
        size_t batch = X.rows();
        if (h_state.rows() != (int)batch) {
            h_state = Matrix<T>(batch, hidden_size);
            c_state = Matrix<T>(batch, hidden_size);
            std::fill(h_state.Flatten(), h_state.Flatten() + (batch * hidden_size), 0);
            std::fill(c_state.Flatten(), c_state.Flatten() + (batch * hidden_size), 0);
        }

        StepCache cache;
        cache.x = X;
        cache.prev_h = h_state;
        cache.prev_c = c_state;
        cache.gates = Matrix<T>(batch, 4 * hidden_size);
        cache.tanh_c = Matrix<T>(batch, hidden_size);

        Matrix<T> Z(batch, 4 * hidden_size), Zh(batch, 4 * hidden_size);
        this->solver_->multiply(batch, 4 * hidden_size, input_size, X.Flatten(), W_all.Flatten(), Z.Flatten());
        this->solver_->multiply(batch, 4 * hidden_size, hidden_size, h_state.Flatten(), U_all.Flatten(), Zh.Flatten());

        Matrix<T> next_c(batch, hidden_size), next_h(batch, hidden_size);
        T *g = cache.gates.Flatten(), *z = Z.Flatten(), *zh = Zh.Flatten(), *b = b_all.Flatten();
        T *nc = next_c.Flatten(), *nh = next_h.Flatten(), *pc = cache.prev_c.Flatten(), *tc = cache.tanh_c.Flatten();

        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < (size_t)hidden_size; ++j) {
                size_t g_base = i * 4 * hidden_size + j;
                T f  = sigmoid(z[g_base] + zh[g_base] + b[j]);
                T in = sigmoid(z[g_base + hidden_size] + zh[g_base + hidden_size] + b[j + hidden_size]);
                T cb = std::tanh(z[g_base + 2*hidden_size] + zh[g_base + 2*hidden_size] + b[j + 2*hidden_size]);
                T o  = sigmoid(z[g_base + 3*hidden_size] + zh[g_base + 3*hidden_size] + b[j + 3*hidden_size]);

                g[g_base] = f; g[g_base+hidden_size] = in; g[g_base+2*hidden_size] = cb; g[g_base+3*hidden_size] = o;

                size_t s_idx = i * hidden_size + j;
                nc[s_idx] = f * pc[s_idx] + in * cb;
                
                // Stability clamp for cell state
                nc[s_idx] = std::max(-STATE_LIMIT, std::min(STATE_LIMIT, nc[s_idx]));

                tc[s_idx] = std::tanh(nc[s_idx]);
                nh[s_idx] = o * tc[s_idx];
            }
        }
        
        h_state = next_h;
        c_state = next_c;
        history.push_back(cache);
        return h_state;
    }

    Matrix<T> Backward(const Matrix<T>& grad_out_last) override {
        if (history.empty()) return Matrix<T>(0, 0);

        size_t batch = grad_out_last.rows();
        size_t sequence_length = history.size();

        Matrix<T> dW_total(input_size, 4 * hidden_size);
        Matrix<T> dU_total(hidden_size, 4 * hidden_size);
        Matrix<T> db_total(1, 4 * hidden_size);
        
        // Initialize gradient flows: grad_out_last is only for the final timestep
        Matrix<T> dh_flow(batch, hidden_size);
        Matrix<T> dc_flow(batch, hidden_size);
        
        // Copy grad_out_last to dh_flow for the last timestep
        std::copy(grad_out_last.Flatten(), grad_out_last.Flatten() + batch * hidden_size, 
                  dh_flow.Flatten());
        
        // Initialize dc_flow to zero
        std::fill(dc_flow.Flatten(), dc_flow.Flatten() + batch * hidden_size, 0);

        std::vector<Matrix<T>> dx_sequence(sequence_length);

        for (int t = (int)sequence_length - 1; t >= 0; --t) {
            StepCache& curr = history[t];
            Matrix<T> dZ(batch, 4 * hidden_size);
            
            T *dz = dZ.Flatten(), *g = curr.gates.Flatten(), *pc = curr.prev_c.Flatten(), *tc = curr.tanh_c.Flatten();
            T *dh = dh_flow.Flatten(), *dc = dc_flow.Flatten();

            Matrix<T> dc_curr(batch, hidden_size);  // Current cell state gradient
            T* dc_curr_ptr = dc_curr.Flatten();

            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < (size_t)hidden_size; ++j) {
                    size_t s_idx = i * hidden_size + j;
                    size_t g_idx = i * 4 * hidden_size + j;

                    // Gradient through output gate and tanh
                    T d_tanh_c = safe_dtanh(tc[s_idx]);
                    T dc_from_h = dh[s_idx] * g[g_idx + 3 * hidden_size] * d_tanh_c;
                    
                    // Total cell state gradient: from current hidden state + from previous cell state (via forget gate)
                    T total_dc = dc_from_h + dc[s_idx];
                    total_dc = std::max(static_cast<T>(-10.0), std::min(static_cast<T>(10.0), total_dc));
                    dc_curr_ptr[s_idx] = total_dc;

                    // Gate gradients
                    T forget_gate = g[g_idx];
                    T input_gate = g[g_idx + hidden_size];
                    T cell_candidate = g[g_idx + 2*hidden_size];
                    T output_gate = g[g_idx + 3*hidden_size];

                    // dL/dZ for each gate
                    dz[g_idx] = total_dc * pc[s_idx] * d_sigmoid(forget_gate);  // Forget gate
                    dz[g_idx + hidden_size] = total_dc * cell_candidate * d_sigmoid(input_gate);  // Input gate
                    dz[g_idx + 2*hidden_size] = total_dc * input_gate * safe_dtanh(cell_candidate);  // Cell candidate
                    dz[g_idx + 3*hidden_size] = dh[s_idx] * tc[s_idx] * d_sigmoid(output_gate);  // Output gate
                }
            }

            // Compute weight gradients
            Matrix<T> dW_step(input_size, 4 * hidden_size), dU_step(hidden_size, 4 * hidden_size);
            this->solver_->multiply(input_size, 4 * hidden_size, batch, curr.x.Transpose().Flatten(), dZ.Flatten(), dW_step.Flatten());
            this->solver_->multiply(hidden_size, 4 * hidden_size, batch, curr.prev_h.Transpose().Flatten(), dZ.Flatten(), dU_step.Flatten());

            // Accumulate gradients
            for(size_t i = 0; i < dW_total.rows() * dW_total.cols(); ++i) dW_total.Flatten()[i] += dW_step.Flatten()[i];
            for(size_t i = 0; i < dU_total.rows() * dU_total.cols(); ++i) dU_total.Flatten()[i] += dU_step.Flatten()[i];
            for (size_t i = 0; i < batch; ++i) {
                for (size_t j = 0; j < 4 * (size_t)hidden_size; ++j) db_total.Flatten()[j] += dz[i * 4 * hidden_size + j];
            }

            // Compute input gradient
            dx_sequence[t] = Matrix<T>(batch, input_size);
            this->solver_->multiply(batch, input_size, 4 * hidden_size, dZ.Flatten(), W_all.Transpose().Flatten(), dx_sequence[t].Flatten());
            
            // Compute hidden state gradient for previous timestep
            Matrix<T> dh_prev(batch, hidden_size);
            this->solver_->multiply(batch, hidden_size, 4 * hidden_size, dZ.Flatten(), U_all.Transpose().Flatten(), dh_prev.Flatten());
            
            // Prepare for previous timestep: dh_flow is the gradient to pass back
            dh_flow = dh_prev;
            dc_flow = dc_curr;  // Cell state gradient flows back through forget gate
        }

        clipGradients(dW_total, dU_total, db_total, static_cast<T>(1.0));

        if (this->optimizer_) {
            this->optimizer_->Optimize(W_all, dW_total);
            this->optimizer_->Optimize(U_all, dU_total);
            this->optimizer_->Optimize(b_all, db_total);
        }

        history.clear(); 
        return dx_sequence[0]; 
    }

    T* Serialize() {
        size_t w_len = W_all.rows() * W_all.cols();
        size_t u_len = U_all.rows() * U_all.cols();
        size_t b_len = b_all.rows() * b_all.cols();
        size_t total_len = w_len + u_len + b_len;
        
        T* buffer = new T[total_len + 1];
        buffer[0] = static_cast<T>(total_len);
        
        T* w_data = W_all.Flatten();
        T* u_data = U_all.Flatten();
        T* b_data = b_all.Flatten();
        
        if (!w_data || !u_data || !b_data) {
            std::cerr << "Error: LSTM weight data is null during serialization" << std::endl;
            delete[] buffer;
            return nullptr;
        }
        
        std::copy(w_data, w_data + w_len, buffer + 1);
        std::copy(u_data, u_data + u_len, buffer + 1 + w_len);
        std::copy(b_data, b_data + b_len, buffer + 1 + w_len + u_len);
        return buffer;
    }

    void Deserialize(T* data) {
        if (!data) {
            std::cerr << "Error: LSTM deserialization data is null" << std::endl;
            return;
        }
        
        size_t w_len = W_all.rows() * W_all.cols();
        size_t u_len = U_all.rows() * U_all.cols();
        size_t b_len = b_all.rows() * b_all.cols();
        
        T* w_data = W_all.Flatten();
        T* u_data = U_all.Flatten();
        T* b_data = b_all.Flatten();
        
        if (!w_data || !u_data || !b_data) {
            std::cerr << "Error: Cannot get LSTM weights for deserialization" << std::endl;
            return;
        }
        
        std::copy(data + 1, data + 1 + w_len, w_data);
        std::copy(data + 1 + w_len, data + 1 + w_len + u_len, u_data);
        std::copy(data + 1 + w_len + u_len, data + 1 + w_len + u_len + b_len, b_data);
    }
};

#endif