#ifndef LSTM_HPP
#define LSTM_HPP

#include "Layer.hpp"
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

template <typename T> 
class LSTM : public Layer<T>  {
private:
    int input_features;
    int hidden_size;

    // Persistent states across time steps
    Matrix<T> h_state;      // hidden state
    Matrix<T> c_state;      // cell state

    // Input and intermediate caches
    Matrix<T> input_cache;
    
    // Gate activation caches for backprop
    Matrix<T> cache_f, cache_i, cache_o, cache_c_bar, cache_tanh_c;
    Matrix<T> cache_c_new;  // Cached new cell state
    Matrix<T> prev_h_state, prev_c_state;

    // Weight matrices - Input weights (input_features x hidden_size)
    Matrix<T> W_f, W_i, W_c, W_o;

    // Recurrent weights (hidden_size x hidden_size)
    Matrix<T> U_f, U_i, U_c, U_o;

    // Bias vectors (1 x hidden_size)
    Matrix<T> b_f, b_i, b_c, b_o;

    // BPTT states - for accumulating gradients through time
    Matrix<T> d_c_next;  // Cell state gradient from next time step
    Matrix<T> d_h_next;  // Hidden state gradient to accumulate

    // Helper: Initialize matrix with normal distribution
    void initializeMatrix(Matrix<T>& mat, int fan_in, int fan_out, const WeightInit& technique, std::default_random_engine& generator) {
        T std_dev;
        
        if (technique == WeightInit::Xavier) {
            std_dev = std::sqrt(static_cast<T>(2.0) / (fan_in + fan_out));
        } else { 
            std_dev = std::sqrt(static_cast<T>(2.0) / fan_in);
        }

        std::normal_distribution<T> distribution(static_cast<T>(0.0), std_dev);
        
        T* data = mat.Flatten();
        size_t size = mat.rows() * mat.cols();
        for(size_t i = 0; i < size; ++i) {
            data[i] = distribution(generator);
        }
    }

public: 
    LSTM(const std::shared_ptr<Matrix_Solver<T>>& solver, 
         const std::shared_ptr<Optimizer<T>>& optimizer, 
         int input_features, 
         int hidden_size)
        : Layer<T>(solver, optimizer), input_features(input_features), hidden_size(hidden_size) 
    {
        initMatrices();
    }

    // Initialize matrix dimensions
    void initMatrices() {
        // Input weights (input_features x hidden_size)
        W_f = Matrix<T>(input_features, hidden_size);
        W_i = Matrix<T>(input_features, hidden_size);
        W_c = Matrix<T>(input_features, hidden_size);
        W_o = Matrix<T>(input_features, hidden_size);

        // Recurrent weights (hidden_size x hidden_size)
        U_f = Matrix<T>(hidden_size, hidden_size);
        U_i = Matrix<T>(hidden_size, hidden_size);
        U_c = Matrix<T>(hidden_size, hidden_size);
        U_o = Matrix<T>(hidden_size, hidden_size);

        // Bias (1 x hidden_size)
        b_f = Matrix<T>(1, hidden_size);
        b_i = Matrix<T>(1, hidden_size);
        b_c = Matrix<T>(1, hidden_size);
        b_o = Matrix<T>(1, hidden_size);

        // Initialize states to empty
        h_state = Matrix<T>(0, 0);
        c_state = Matrix<T>(0, 0);
        d_c_next = Matrix<T>(0, 0);
        d_h_next = Matrix<T>(0, 0);
    }

    void resetState() {
        h_state = Matrix<T>(0, 0);
        c_state = Matrix<T>(0, 0);
        d_c_next = Matrix<T>(0, 0);
        d_h_next = Matrix<T>(0, 0);
    }
    
    void WeightInitialization(const WeightInit& technique) override {
        std::random_device rd;
        std::default_random_engine generator(rd());

        // Initialize input weights (W)
        initializeMatrix(W_f, input_features, hidden_size, technique, generator);
        initializeMatrix(W_i, input_features, hidden_size, technique, generator);
        initializeMatrix(W_c, input_features, hidden_size, technique, generator);
        initializeMatrix(W_o, input_features, hidden_size, technique, generator);

        // Initialize recurrent weights (U)
        initializeMatrix(U_f, hidden_size, hidden_size, technique, generator);
        initializeMatrix(U_i, hidden_size, hidden_size, technique, generator);
        initializeMatrix(U_c, hidden_size, hidden_size, technique, generator);
        initializeMatrix(U_o, hidden_size, hidden_size, technique, generator);

        // Bias initialized to zero (default)
    }

    // Activation functions with numerical stability
    inline T sigmoid(T x) { 
        x = std::max(static_cast<T>(-20.0), std::min(static_cast<T>(20.0), x));
        return 1.0 / (1.0 + std::exp(-x)); 
    }
    
    inline T tanh_act(T x) { 
        return std::tanh(x); 
    }
    
    inline T d_sigmoid(T y) { 
        return y * (1.0 - y); 
    }
    
    inline T d_tanh(T y) { 
        return 1.0 - (y * y); 
    }

    // FORWARD PASS - Single time step
    Matrix<T> Forward(const Matrix<T>& X) override {
        size_t batch_size = X.rows();

        input_cache = X;

        // Initialize or reinitialize states if needed
        if ((int)batch_size != h_state.rows() || hidden_size != h_state.cols()) {
            h_state = Matrix<T>(batch_size, hidden_size); 
            c_state = Matrix<T>(batch_size, hidden_size);
        }

        prev_h_state = h_state; 
        prev_c_state = c_state;

        Matrix<T> next_h(batch_size, hidden_size);
        Matrix<T> next_c(batch_size, hidden_size);

        // Cache matrices for backward pass
        cache_f = Matrix<T>(batch_size, hidden_size);
        cache_i = Matrix<T>(batch_size, hidden_size);
        cache_c_bar = Matrix<T>(batch_size, hidden_size);
        cache_o = Matrix<T>(batch_size, hidden_size);
        cache_tanh_c = Matrix<T>(batch_size, hidden_size);
        cache_c_new = Matrix<T>(batch_size, hidden_size);

        // Temp matrices for gate pre-activations
        Matrix<T> XW_f(batch_size, hidden_size), hU_f(batch_size, hidden_size);
        Matrix<T> XW_i(batch_size, hidden_size), hU_i(batch_size, hidden_size);
        Matrix<T> XW_c(batch_size, hidden_size), hU_c(batch_size, hidden_size);
        Matrix<T> XW_o(batch_size, hidden_size), hU_o(batch_size, hidden_size);

        const T* x_ptr = X.Flatten();
        const T* h_ptr = prev_h_state.Flatten();
        
        // Matrix multiplications
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_f.Flatten(), XW_f.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_f.Flatten(), hU_f.Flatten());

        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_i.Flatten(), XW_i.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_i.Flatten(), hU_i.Flatten());

        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_c.Flatten(), XW_c.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_c.Flatten(), hU_c.Flatten());

        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_o.Flatten(), XW_o.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_o.Flatten(), hU_o.Flatten());

        const T* r_xw_f = XW_f.Flatten(); const T* r_hu_f = hU_f.Flatten(); const T* b_f_p = b_f.Flatten();
        const T* r_xw_i = XW_i.Flatten(); const T* r_hu_i = hU_i.Flatten(); const T* b_i_p = b_i.Flatten();
        const T* r_xw_c = XW_c.Flatten(); const T* r_hu_c = hU_c.Flatten(); const T* b_c_p = b_c.Flatten();
        const T* r_xw_o = XW_o.Flatten(); const T* r_hu_o = hU_o.Flatten(); const T* b_o_p = b_o.Flatten();
        
        T* nh_ptr = next_h.Flatten();
        T* nc_ptr = next_c.Flatten();
        const T* pc_ptr = prev_c_state.Flatten();

        T* cf = cache_f.Flatten(); 
        T* ci = cache_i.Flatten();
        T* cc = cache_c_bar.Flatten(); 
        T* co = cache_o.Flatten();
        T* ctc = cache_tanh_c.Flatten();
        T* ccn = cache_c_new.Flatten();

        size_t total = batch_size * hidden_size;

        // Compute gates and states
        for (size_t idx = 0; idx < total; ++idx) {
            size_t col = idx % hidden_size;

            T f = sigmoid(r_xw_f[idx] + r_hu_f[idx] + b_f_p[col]);
            T i = sigmoid(r_xw_i[idx] + r_hu_i[idx] + b_i_p[col]);
            T c_bar = tanh_act(r_xw_c[idx] + r_hu_c[idx] + b_c_p[col]);
            T o = sigmoid(r_xw_o[idx] + r_hu_o[idx] + b_o_p[col]);

            cf[idx] = f; 
            ci[idx] = i; 
            cc[idx] = c_bar; 
            co[idx] = o;

            T old_c = pc_ptr[idx];
            T new_c = (f * old_c) + (i * c_bar);
            nc_ptr[idx] = new_c;
            ccn[idx] = new_c;

            T tanh_new_c = tanh_act(new_c);
            ctc[idx] = tanh_new_c;
            nh_ptr[idx] = o * tanh_new_c;
        }

        h_state = next_h;
        c_state = next_c;

        return h_state;
    }
    
    // BACKWARD PASS - Full BPTT (Backpropagation Through Time)
    // Properly accumulates gradients through cell state across time steps
    Matrix<T> Backward(const Matrix<T>& grad_output) override {
        size_t batch_size = grad_output.rows();
        size_t total_elem = batch_size * hidden_size;

        // Initialize or allocate BPTT gradient accumulators
        if (d_c_next.rows() != (int)batch_size || d_c_next.cols() != hidden_size) {
            d_c_next = Matrix<T>(batch_size, hidden_size);  // Gradient from next time step's cell state
            d_h_next = Matrix<T>(batch_size, hidden_size);  // Gradient from next time step's hidden state
        }

        // Gate gradient matrices
        Matrix<T> d_f(batch_size, hidden_size);
        Matrix<T> d_i(batch_size, hidden_size);
        Matrix<T> d_c_bar(batch_size, hidden_size);
        Matrix<T> d_o(batch_size, hidden_size);
        Matrix<T> d_c(batch_size, hidden_size);
        Matrix<T> d_h(batch_size, hidden_size);

        const T* gh_ptr = grad_output.Flatten();
        const T* tanh_c_ptr = cache_tanh_c.Flatten();
        const T* o_ptr = cache_o.Flatten();
        const T* f_ptr = cache_f.Flatten();
        const T* i_ptr = cache_i.Flatten();
        const T* c_bar_ptr = cache_c_bar.Flatten();
        const T* prev_c_ptr = prev_c_state.Flatten();
        const T* dc_next_ptr = d_c_next.Flatten();
        const T* dh_next_ptr = d_h_next.Flatten();

        T* df = d_f.Flatten();
        T* di = d_i.Flatten();
        T* dc_bar = d_c_bar.Flatten();
        T* doo = d_o.Flatten();
        T* dc = d_c.Flatten();
        T* dh = d_h.Flatten();

        // Compute gate gradients with BPTT
        for (size_t k = 0; k < total_elem; ++k) {
            // Gradient w.r.t. hidden state: from current output + accumulated from next time step
            dh[k] = gh_ptr[k] + dh_next_ptr[k];
            
            T tc = tanh_c_ptr[k];
            T o_val = o_ptr[k];

            // Output gate gradient
            doo[k] = dh[k] * tc * d_sigmoid(o_val);

            // Cell state gradient: from current hidden state + from next time step
            T d_c_from_h = dh[k] * o_val * d_tanh(tc);
            dc[k] = d_c_from_h + dc_next_ptr[k];

            // Candidate cell gradient
            T i_val = i_ptr[k];
            T c_bar_val = c_bar_ptr[k];
            dc_bar[k] = (dc[k] * i_val) * d_tanh(c_bar_val);

            // Input gate gradient
            di[k] = (dc[k] * c_bar_val) * d_sigmoid(i_val);

            // Forget gate gradient
            T prev_c_val = prev_c_ptr[k];
            df[k] = (dc[k] * prev_c_val) * d_sigmoid(f_ptr[k]);
        }

        // Initialize input gradient
        Matrix<T> dX(batch_size, input_features);

        Matrix<T> X_T = input_cache.Transpose();
        Matrix<T> H_T = prev_h_state.Transpose();

        // Apply gradient updates for each gate
        auto apply_gate_backward = [&](Matrix<T>& delta, Matrix<T>& W, Matrix<T>& U, Matrix<T>& b) {
            // Weight gradient: dW = X^T * delta
            Matrix<T> dW(input_features, hidden_size);
            this->solver_->multiply(input_features, hidden_size, batch_size, 
                                    X_T.Flatten(), delta.Flatten(), dW.Flatten());

            // Recurrent weight gradient: dU = H_prev^T * delta
            Matrix<T> dU(hidden_size, hidden_size);
            this->solver_->multiply(hidden_size, hidden_size, batch_size, 
                                    H_T.Flatten(), delta.Flatten(), dU.Flatten());

            // Bias gradient: db = sum over batch
            Matrix<T> db(1, hidden_size);
            T* db_ptr = db.Flatten();
            const T* delta_ptr = delta.Flatten();
            
            for(int j = 0; j < hidden_size; ++j) {
                db_ptr[j] = 0;
            }
            
            for (size_t r = 0; r < batch_size; ++r) {
                for (size_t c = 0; c < (size_t)hidden_size; ++c) {
                    db_ptr[c] += delta_ptr[r * hidden_size + c];
                }
            }

            // Input gradient contribution: dX += delta * W^T
            Matrix<T> W_T = W.Transpose();
            Matrix<T> dXi(batch_size, input_features);
            this->solver_->multiply(batch_size, input_features, hidden_size, 
                                    delta.Flatten(), W_T.Flatten(), dXi.Flatten());

            // Accumulate
            T* dx_raw = dX.Flatten();
            const T* dxi_raw = dXi.Flatten();
            size_t dx_size = batch_size * input_features;
            for(size_t i = 0; i < dx_size; ++i) {
                dx_raw[i] += dxi_raw[i];
            }

            // Update parameters
            if(this->optimizer_) {
                this->optimizer_->Optimize(W, dW);
                this->optimizer_->Optimize(U, dU);
                this->optimizer_->Optimize(b, db);
            }
        };

        // Apply backward for all gates
        apply_gate_backward(d_f, W_f, U_f, b_f);
        apply_gate_backward(d_i, W_i, U_i, b_i);
        apply_gate_backward(d_c_bar, W_c, U_c, b_c);
        apply_gate_backward(d_o, W_o, U_o, b_o);

        // Compute recurrent gradients for next iteration of BPTT
        // d_h_next will be used by the previous time step
        Matrix<T> d_h_recurrent(batch_size, hidden_size);
        
        // Contribution from forget gate: d_h_prev += d_f * U_f^T
        Matrix<T> U_f_T = U_f.Transpose();
        this->solver_->multiply(batch_size, hidden_size, hidden_size,
                                d_f.Flatten(), U_f_T.Flatten(), d_h_recurrent.Flatten());
        
        // Add contributions from other gates
        Matrix<T> temp(batch_size, hidden_size);
        
        Matrix<T> U_i_T = U_i.Transpose();
        this->solver_->multiply(batch_size, hidden_size, hidden_size,
                                d_i.Flatten(), U_i_T.Flatten(), temp.Flatten());
        T* dhr_ptr = d_h_recurrent.Flatten();
        const T* temp_ptr = temp.Flatten();
        for (size_t i = 0; i < batch_size * hidden_size; ++i) {
            dhr_ptr[i] += temp_ptr[i];
        }
        
        Matrix<T> U_c_T = U_c.Transpose();
        this->solver_->multiply(batch_size, hidden_size, hidden_size,
                                d_c_bar.Flatten(), U_c_T.Flatten(), temp.Flatten());
        for (size_t i = 0; i < batch_size * hidden_size; ++i) {
            dhr_ptr[i] += temp_ptr[i];
        }
        
        Matrix<T> U_o_T = U_o.Transpose();
        this->solver_->multiply(batch_size, hidden_size, hidden_size,
                                d_o.Flatten(), U_o_T.Flatten(), temp.Flatten());
        for (size_t i = 0; i < batch_size * hidden_size; ++i) {
            dhr_ptr[i] += temp_ptr[i];
        }

        // Store gradients for next time step (BPTT)
        // These will be added to the gradient of the previous hidden state
        d_h_next = d_h_recurrent;
        
        // Cell state gradient for forget gate: d_c_prev = d_c * f
        const T* f_cache_ptr = cache_f.Flatten();
        const T* dc_ptr = d_c.Flatten();
        T* dcn_ptr = d_c_next.Flatten();
        for (size_t i = 0; i < batch_size * hidden_size; ++i) {
            dcn_ptr[i] = dc_ptr[i] * f_cache_ptr[i];
        }

        return dX;
    }

    // Helper to reset BPTT accumulators for a new sequence
    void resetBPTTGradients() {
        if (d_c_next.rows() > 0) {
            T* dc_ptr = d_c_next.Flatten();
            T* dh_ptr = d_h_next.Flatten();
            size_t total = d_c_next.rows() * d_c_next.cols();
            for (size_t i = 0; i < total; ++i) {
                dc_ptr[i] = 0;
                dh_ptr[i] = 0;
            }
        }
    }

    T* Serialize() {
        // Total size: 4 bytes for count + data for all weights
        // Weights: W_f, W_i, W_c, W_o (4*input_features*hidden_size)
        // + U_f, U_i, U_c, U_o (4*hidden_size*hidden_size)
        // + b_f, b_i, b_c, b_o (4*hidden_size)
        size_t w_size = input_features * hidden_size;
        size_t u_size = hidden_size * hidden_size;
        size_t b_size = hidden_size;
        size_t total_elements = 4*w_size + 4*u_size + 4*b_size;
        
        T* buffer = new T[total_elements + 1];
        buffer[0] = static_cast<T>(total_elements); // Store total count
        
        size_t idx = 1;
        
        // Copy W matrices
        const T* w_f = W_f.Flatten();
        const T* w_i = W_i.Flatten();
        const T* w_c = W_c.Flatten();
        const T* w_o = W_o.Flatten();
        
        for(size_t i = 0; i < w_size; ++i) buffer[idx++] = w_f[i];
        for(size_t i = 0; i < w_size; ++i) buffer[idx++] = w_i[i];
        for(size_t i = 0; i < w_size; ++i) buffer[idx++] = w_c[i];
        for(size_t i = 0; i < w_size; ++i) buffer[idx++] = w_o[i];
        
        // Copy U matrices
        const T* u_f = U_f.Flatten();
        const T* u_i = U_i.Flatten();
        const T* u_c = U_c.Flatten();
        const T* u_o = U_o.Flatten();
        
        for(size_t i = 0; i < u_size; ++i) buffer[idx++] = u_f[i];
        for(size_t i = 0; i < u_size; ++i) buffer[idx++] = u_i[i];
        for(size_t i = 0; i < u_size; ++i) buffer[idx++] = u_c[i];
        for(size_t i = 0; i < u_size; ++i) buffer[idx++] = u_o[i];
        
        // Copy bias vectors
        const T* b_f_p = b_f.Flatten();
        const T* b_i_p = b_i.Flatten();
        const T* b_c_p = b_c.Flatten();
        const T* b_o_p = b_o.Flatten();
        
        for(size_t i = 0; i < b_size; ++i) buffer[idx++] = b_f_p[i];
        for(size_t i = 0; i < b_size; ++i) buffer[idx++] = b_i_p[i];
        for(size_t i = 0; i < b_size; ++i) buffer[idx++] = b_c_p[i];
        for(size_t i = 0; i < b_size; ++i) buffer[idx++] = b_o_p[i];
        
        return buffer;
    }

    void Deserialize(T* data) {
        if (!data) return;
        
        size_t w_size = input_features * hidden_size;
        size_t u_size = hidden_size * hidden_size;
        size_t b_size = hidden_size;
        size_t idx = 1; // Skip count at index 0
        
        // Restore W matrices
        T* w_f = W_f.Flatten();
        T* w_i = W_i.Flatten();
        T* w_c = W_c.Flatten();
        T* w_o = W_o.Flatten();
        
        for(size_t i = 0; i < w_size; ++i) w_f[i] = data[idx++];
        for(size_t i = 0; i < w_size; ++i) w_i[i] = data[idx++];
        for(size_t i = 0; i < w_size; ++i) w_c[i] = data[idx++];
        for(size_t i = 0; i < w_size; ++i) w_o[i] = data[idx++];
        
        // Restore U matrices
        T* u_f = U_f.Flatten();
        T* u_i = U_i.Flatten();
        T* u_c = U_c.Flatten();
        T* u_o = U_o.Flatten();
        
        for(size_t i = 0; i < u_size; ++i) u_f[i] = data[idx++];
        for(size_t i = 0; i < u_size; ++i) u_i[i] = data[idx++];
        for(size_t i = 0; i < u_size; ++i) u_c[i] = data[idx++];
        for(size_t i = 0; i < u_size; ++i) u_o[i] = data[idx++];
        
        // Restore bias vectors
        T* b_f_p = b_f.Flatten();
        T* b_i_p = b_i.Flatten();
        T* b_c_p = b_c.Flatten();
        T* b_o_p = b_o.Flatten();
        
        for(size_t i = 0; i < b_size; ++i) b_f_p[i] = data[idx++];
        for(size_t i = 0; i < b_size; ++i) b_i_p[i] = data[idx++];
        for(size_t i = 0; i < b_size; ++i) b_c_p[i] = data[idx++];
        for(size_t i = 0; i < b_size; ++i) b_o_p[i] = data[idx++];
    }
};

#endif
    
    

       
        