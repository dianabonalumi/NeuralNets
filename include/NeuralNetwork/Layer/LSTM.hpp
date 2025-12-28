#ifndef LSTM_HPP
#define LSTM_HPP

#include "Layer.hpp"
#include <cmath>
#include <vector>
#include <random>

template <typename T> 
class LSTM : public Layer<T> >{
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

    //FORWARD PASS
    Matrix<T> Forward(const Matrix<T>& X) override {
        size_t batch_size = X.rows();

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

        // Forget Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_f.Flatten(), XW_f.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_f.Flatten(), hU_f.Flatten());

        // Input Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_i.Flatten(), XW_i.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_i.Flatten(), hU_i.Flatten());

        //  Candidate (Cell) Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_c.Flatten(), XW_c.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_c.Flatten(), hU_c.Flatten());

        // Output Gate
        this->solver_->multiply(batch_size, hidden_size, input_features, x_ptr, W_o.Flatten(), XW_o.Flatten());
        this->solver_->multiply(batch_size, hidden_size, hidden_size, h_ptr, U_o.Flatten(), hU_o.Flatten());

        //Calcolo delle somme e attivazioni
        
        // Accediamo ai dati grezzi per velocità
        const T* raw_XW_f = XW_f.Flatten(); const T* raw_hU_f = hU_f.Flatten();
        const T* raw_XW_i = XW_i.Flatten(); const T* raw_hU_i = hU_i.Flatten();
        const T* raw_XW_c = XW_c.Flatten(); const T* raw_hU_c = hU_c.Flatten();
        const T* raw_XW_o = XW_o.Flatten(); const T* raw_hU_o = hU_o.Flatten();

        // Dati dei bias (attenzione: bias è 1xHidden, va broadcastato su tutte le righe)
        const T* b_f_ptr = b_f.Flatten();
        const T* b_i_ptr = b_i.Flatten();
        const T* b_c_ptr = b_c.Flatten();
        const T* b_o_ptr = b_o.Flatten();

        size_t total_elements = batch_size * hidden_size;

        // Usiamo un ciclo lineare unico (più cache friendly del doppio for)
        
        
        // Punter alle matrici output/cache
        T* next_h_ptr = next_h.Flatten();
        T* next_c_ptr = next_c.Flatten();
        const T* prev_c_ptr = prev_c_state.Flatten();
        
        // Puntatori cache
        T* c_f_ptr = cache_f.Flatten();
        T* c_i_ptr = cache_i.Flatten();
        T* c_cb_ptr = cache_c_bar.Flatten();
        T* c_o_ptr = cache_o.Flatten();
        T* c_tc_ptr = cache_tanh_c.Flatten();

        for (size_t idx = 0; idx < total_elements; ++idx) {
            size_t c = idx % hidden_size; // Colonna corrente (per il bias)

            // Calcolo Somme (XW + hU + b)
            T val_f = raw_XW_f[idx] + raw_hU_f[idx] + b_f_ptr[c];
            T val_i = raw_XW_i[idx] + raw_hU_i[idx] + b_i_ptr[c];
            T val_c = raw_XW_c[idx] + raw_hU_c[idx] + b_c_ptr[c];
            T val_o = raw_XW_o[idx] + raw_hU_o[idx] + b_o_ptr[c];

            // Attivazioni
            T f = sigmoid(val_f);       // Forget Gate
            T i = sigmoid(val_i);       // Input Gate
            T c_bar = tanh_act(val_c);  // Candidate
            T o = sigmoid(val_o);       // Output Gate

            // Salvataggio Cache
            c_f_ptr[idx] = f;
            c_i_ptr[idx] = i;
            c_cb_ptr[idx] = c_bar;
            c_o_ptr[idx] = o;

            // Aggiornamento Stati
            // C_t = f * C_{t-1} + i * c_bar
            T old_C = prev_c_ptr[idx];
            T new_C = (f * old_C) + (i * c_bar);
            next_c_ptr[idx] = new_C;

            // h_t = o * tanh(C_t)
            T tanh_new_C = tanh_act(new_C);
            T new_h = o * tanh_new_C;
            next_h_ptr[idx] = new_h;
            
            c_tc_ptr[idx] = tanh_new_C;
        }

        // Aggiorna stati interni
        h_state = next_h;
        c_state = next_c;

        return h_state;
    }
}

       
        