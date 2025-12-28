#ifndef LSTM_HPP
#define LSTM_HPP

#include "Layer.hpp"
#include <cmath>
#include <vector>
#include <cstring> //se necessario per memset

template <typename T> 
class LSTM : public Layer<T>  {
private: 
    int input_features;
    int hidden_size;

    //definisco gli stati interni

    Matrix<T> h_state; // hidden state, memoria a breve termine
    Matrix<T> c_state; // Cell state, memoria a lungo termine

    // Pesi e bias: W sono i pesi di input, U i pesi ricorrenti, b bias
    
    //Forget Gate
    Matrix<T> W_f, U_f, b_f;

    //Input Gate
    Matrix<T> W_i, U_i, b_i;

    //Cell gate
    Matrix<T> W_c, U_c, b_c;

    //Output gate
    Matrix<T> W_o, U_o, b_o;

    //Cache per backpropagation
    Matrix<T> cache_f, cache_i, cache_o, cache_c_bar, cache_tanh_c;
    Matrix<T> prev_h_state, prev_c_state;

    public: 
    LSTM(const std::shared_ptr<Matrix_Solver<T>>& solver, 
         const std::shared_ptr<Optimizer<T>>& optimizer, 
         int input_features, 
         int hidden_size)
        : Layer<T>(solver, optimizer), input_features(input_features), hidden_size(hidden_size) 
    {
        // Inizializza pesi e stati
        initMatrices();
    }

    // Inizializzazione le dimensioni delle matrici, allocazione delo spazio per tutte le matrici create prima
    void initMatrices() {
        // Pesi Input (input_features x hidden_size)
        W_f = Matrix<T>(input_features, hidden_size);
        W_i = Matrix<T>(input_features, hidden_size);
        W_c = Matrix<T>(input_features, hidden_size);
        W_o = Matrix<T>(input_features, hidden_size);

        // Pesi Ricorrenti (hidden_size x hidden_size)
        U_f = Matrix<T>(hidden_size, hidden_size);
        U_i = Matrix<T>(hidden_size, hidden_size);
        U_c = Matrix<T>(hidden_size, hidden_size);
        U_o = Matrix<T>(hidden_size, hidden_size);

        // Bias (1 x hidden_size) - Broadcasted durante il calcolo
        b_f = Matrix<T>(1, hidden_size);
        b_i = Matrix<T>(1, hidden_size);
        b_c = Matrix<T>(1, hidden_size);
        b_o = Matrix<T>(1, hidden_size);

        // Gli stati iniziali sono a 0.
        h_state = Matrix<T>(0, 0);
        c_state = Matrix<T>(0, 0);
    }

    // Ogni volta che devo iniziare una nuova epoca faccio il reset dello stato
    void resetState() {
        // Mettiamo le dimensioni a 0 per forzare la ri-inizializzazione nel Forward
        h_state = Matrix<T>(0, 0);
        c_state = Matrix<T>(0, 0);
    }

    // Funzioni di attivazione helper (inline per brevità)
    T sigmoid(T x) { return 1.0 / (1.0 + std::exp(-x)); }
    T tanh_act(T x) { return std::tanh(x); }
    inline T d_sigmoid(T y) { return y * (1.0 - y); }
    inline T d_tanh(T y) { return 1.0 - (y * y); }

    //FORWARD PASS
    Matrix<T> Forward(const Matrix<T>& X) override {
        size_t batch_size = X.rows();

        input_cache = X; //salvo input per backward

        //Inizializzazione stati se è il primo step o se è cambiato il batch size
        if (h_state.rows() != batch_size || h_state.cols() != hidden_size) {
            h_state = Matrix<T>(batch_size, hidden_size); 
            c_state = Matrix<T>(batch_size, hidden_size);
        }

        // Salviamo gli stati precedenti per il backward
        prev_h_state = h_state; 
        prev_c_state = c_state;

        // Matrice risultato per il nuovo hidden state
        Matrix<T> next_h(batch_size, hidden_size);
        Matrix<T> next_c(batch_size, hidden_size);

        // matrici cache per il backward
        cache_f = Matrix<T>(batch_size, hidden_size);
        cache_i = Matrix<T>(batch_size, hidden_size);
        cache_c_bar = Matrix<T>(batch_size, hidden_size);
        cache_o = Matrix<T>(batch_size, hidden_size);
        cache_tanh_c = Matrix<T>(batch_size, hidden_size);

        //Calcolo di X*W e h*U per tutte e 4 le porte.

        // Matrici temporanee per i risultati intermedi
        Matrix<T> XW_f(batch_size, hidden_size), hU_f(batch_size, hidden_size);
        Matrix<T> XW_i(batch_size, hidden_size), hU_i(batch_size, hidden_size);
        Matrix<T> XW_c(batch_size, hidden_size), hU_c(batch_size, hidden_size);
        Matrix<T> XW_o(batch_size, hidden_size), hU_o(batch_size, hidden_size);

        // Flattening per passare i puntatori al solver
        const T* x_ptr = X.Flatten();
        const T* h_ptr = prev_h_state.Flatten();
        
        //Moltiplicazioni tramite solver

        // Forget Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_f.Flatten(), XW_f.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_f.Flatten(), hU_f.Flatten());

        // Input Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_i.Flatten(), XW_i.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_i.Flatten(), hU_i.Flatten());

        // Cell Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_c.Flatten(), XW_c.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_c.Flatten(), hU_c.Flatten());

        // Output Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_o.Flatten(), XW_o.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_o.Flatten(), hU_o.Flatten());

        //Calcolo delle somme e attivazioni
        
        // Accediamo ai dati grezzi per velocità, puntatori raw per loop combinato
        const T* r_xw_f = XW_f.Flatten(); const T* r_hu_f = hU_f.Flatten(); const T* b_f_p = b_f.Flatten();
        const T* r_xw_i = XW_i.Flatten(); const T* r_hu_i = hU_i.Flatten(); const T* b_i_p = b_i.Flatten();
        const T* r_xw_c = XW_c.Flatten(); const T* r_hu_c = hU_c.Flatten(); const T* b_c_p = b_c.Flatten();
        const T* r_xw_o = XW_o.Flatten(); const T* r_hu_o = hU_o.Flatten(); const T* b_o_p = b_o.Flatten();
        
        
        T* nh_ptr = next_h.Flatten();
        T* nc_ptr = next_c.Flatten();
        const T* pc_ptr = prev_c_state.Flatten(); // Prev C state

        // Puntatori cache
        T* cf = cache_f.Flatten(); T* ci = cache_i.Flatten();
        T* cc = cache_c_bar.Flatten(); T* co = cache_o.Flatten();
        T* ctc = cache_tanh_c.Flatten();

        size_t total = batch_size * hidden_size;

        for (size_t idx = 0; idx < total; ++idx) {
            size_t col = idx % hidden_size;

            T f = sigmoid(r_xw_f[idx] + r_hu_f[idx] + b_f_p[col]);
            T i = sigmoid(r_xw_i[idx] + r_hu_i[idx] + b_i_p[col]);
            T c_bar = tanh_act(r_xw_c[idx] + r_hu_c[idx] + b_c_p[col]);
            T o = sigmoid(r_xw_o[idx] + r_hu_o[idx] + b_o_p[col]);

            // Cache
            cf[idx] = f; ci[idx] = i; cc[idx] = c_bar; co[idx] = o;

            // Stati
            T old_C = pc_ptr[idx];
            T new_C = (f * old_C) + (i * c_bar);
            nc_ptr[idx] = new_C;

            T tanh_new_C = tanh_act(new_C);
            ctc[idx] = tanh_new_C;
            nh_ptr[idx] = o * tanh_new_C;
        }

        h_state = next_h;
        c_state = next_c;

        return h_state;
    }
    
    
    // BACKWARD PASS
    Matrix<T> Backward(const Matrix<T>& grad_output) override {
        size_t batch_size = grad_output.rows();
        size_t total_elem = batch_size * hidden_size;

        //Allocazione Delta
        Matrix<T> d_f(batch_size, hidden_size);
        Matrix<T> d_i(batch_size, hidden_size);
        Matrix<T> d_c_bar(batch_size, hidden_size);
        Matrix<T> d_o(batch_size, hidden_size);

        // Pointers
        const T* gh_ptr = grad_output.Flatten();
        const T* tanh_c_ptr = cache_tanh_c.Flatten();
        const T* o_ptr = cache_o.Flatten();
        const T* f_ptr = cache_f.Flatten();
        const T* i_ptr = cache_i.Flatten();
        const T* c_bar_ptr = cache_c_bar.Flatten();
        const T* prev_c_ptr = prev_c_state.Flatten();

        T* df = d_f.Flatten();
        T* di = d_i.Flatten();
        T* dc = d_c_bar.Flatten(); // dc è d_c_bar
        T* doo = d_o.Flatten();

        // Calcolo Gradienti delle Porte
        for (size_t k = 0; k < total_elem; ++k) {
            T dh = gh_ptr[k];
            T tc = tanh_c_ptr[k];
            T o_val = o_ptr[k];

            // dL / dOutputGate
            T d_o_val = dh * tc * d_sigmoid(o_val);
            doo[k] = d_o_val;

            // dL / dC_state
            T d_C_val = dh * o_val * d_tanh(tc);

            // dL / dCandidate
            T i_val = i_ptr[k];
            T c_bar_val = c_bar_ptr[k];
            dc[k] = (d_C_val * i_val) * d_tanh(c_bar_val); // Nota: derivata tanh(c_bar)

            // dL / dInputGate
            di[k] = (d_C_val * c_bar_val) * d_sigmoid(i_val);

            // dL / dForgetGate
            T prev_c_val = prev_c_ptr[k];
            df[k] = (d_C_val * prev_c_val) * d_sigmoid(f_ptr[k]);
        }

        // Calcolo Gradienti Pesi e Accumulo dX
        
        // Risultato finale dX (Inizializzato a 0)
        Matrix<T> dX(batch_size, input_features);
        
        
        Matrix<T> X_T = input_cache.Transpose();
        
        
        Matrix<T> H_T = prev_h_state.Transpose();

        // Lambda per gestire l'aggiornamento di una singola porta
        auto apply_gate_backward = [&](Matrix<T>& delta, Matrix<T>& W, Matrix<T>& U, Matrix<T>& b) {
            // Calcolo dW = X^T * delta
            // X^T (Feat x Batch) * delta (Batch x Hid) = dW (Feat x Hid)
            Matrix<T> dW(input_features, hidden_size);
            this->solver_->multiply(input_features, hidden_size, batch_size, 
                                    X_T.Flatten(), delta.Flatten(), dW.Flatten());

            // Calcolo dU = H_prev^T * delta
            // H^T (Hid x Batch) * delta (Batch x Hid) = dU (Hid x Hid)
            Matrix<T> dU(hidden_size, hidden_size);
            this->solver_->multiply(hidden_size, hidden_size, batch_size, 
                                    H_T.Flatten(), delta.Flatten(), dU.Flatten());

            // Calcolo db (Somma delle righe di delta)
            Matrix<T> db(1, hidden_size);
            T* db_ptr = db.Flatten();
            const T* delta_ptr = delta.Flatten();
            // Inizializza a 0
            for(int j=0; j<hidden_size; ++j) db_ptr[j] = 0;
            
            // Somma
            for (size_t r = 0; r < batch_size; ++r) {
                for (size_t c = 0; c < hidden_size; ++c) {
                    db_ptr[c] += delta_ptr[r * hidden_size + c];
                }
            }

            // Accumulo Gradiente Input dX += delta * W^T
            // delta (Batch x Hid) * W^T (Hid x Feat) = dXi (Batch x Feat)
            Matrix<T> W_T = W.Transpose();
            Matrix<T> dXi(batch_size, input_features);
            this->solver_->multiply(batch_size, input_features, hidden_size, 
                                    delta.Flatten(), W_T.Flatten(), dXi.Flatten());

            // Somma manuale dX = dX + dXi
            T* dx_raw = dX.Flatten();
            const T* dxi_raw = dXi.Flatten();
            size_t dx_size = batch_size * input_features;
            for(size_t i=0; i<dx_size; ++i) {
                dx_raw[i] += dxi_raw[i];
            }

            // Update pesi usando l'Optimizer
            this->optimizer_->update(W, dW);
            this->optimizer_->update(U, dU);
            this->optimizer_->update(b, db);
        };

        // Applica a tutte le porte
        apply_gate_backward(d_f, W_f, U_f, b_f);
        apply_gate_backward(d_i, W_i, U_i, b_i);
        apply_gate_backward(d_c_bar, W_c, U_c, b_c);
        apply_gate_backward(d_o, W_o, U_o, b_o);

        // N.B.: X_T, H_T e le W_T temporanee vengono distrutte automaticamente qui.

        return dX;
    }
};

#endif
    
    

       
        