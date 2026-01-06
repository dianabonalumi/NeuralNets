#ifndef ENCODER_HPP
#define ENCODER_HPP

#include "Architecture.hpp"
#include "../Layer/Layer.hpp"
#include "../Loss/Loss.hpp"
#include "../Matrix.hpp"
#include "../../matrix_solver.hpp"
#include "../Optimizer/Optimizer.hpp"

#include <vector>
#include <memory>

template <typename T>
class Encoder : public Architecture<T> {
private:
    const std::shared_ptr<Matrix_Solver<T>> solver_;
    const std::shared_ptr<Optimizer<T>> optimizer_;
    const std::shared_ptr<Loss<T>> loss_;
    const int window_, stride_;
    const int in_shape_, out_shape_;
    
    std::shared_ptr<Layer<T>> lstm_layer_; /// !!
    std::shared_ptr<Layer<T>> dense_layer_; /// !!
public:
    Encoder(const std::shared_ptr<Matrix_Solver<T>>& solver,
        const std::shared_ptr<Optimizer<T>>& optimizer,
        const int& window,
        const int& stride,
        const int& in_shape,
        const int& out_shape) 
        : solver_(solver), optimizer_(optimizer), in_shape_(in_shape), out_shape_(out_shape),
        window_(window), stride_(stride) {
    }
    

    //aggiungo metodo per impostare i layer LSTM e Dense
    void setLayers(std::shared_ptr<Layer<T>> lstm, std::shared_ptr<Layer<T>> dense) {
        lstm_layer_ = lstm;
        dense_layer_ = dense;
    }
    
    //aggiungo metodo per estrarre finestra
    Matrix<T> getWindow(const Matrix<T>& full_signal, int start_idx) {
        Matrix<T> window_mat(window_, 1); 
        
        const T* src = full_signal.Flatten();
        T* dst = window_mat.Flatten();
        
        for(int i=0; i<window_; ++i) {
            if(start_idx + i < in_shape_) {    ///!!
                dst[i] = src[start_idx + i];
            } else {
                dst[i] = 0; 
            }
        }
        return window_mat; 
    }


    Matrix<T> Predict(const Matrix<T>& X) {

        // calcolo il numero di finestre
        int num_windows = (in_shape - window_) / stride_ + 1;
        if (num_windows <= 0) num_windows = 1;
        
        
        Matrix<T> latent_output(num_windows, out_shape_);
        T* out_ptr = latent_output.Flatten();

        // faccio cast dei layer per poi chiamarli una volta sola
        auto lstm_ptr = std::dynamic_pointer_cast<LSTM<T>>(lstm_layer_);

        int window_idx = 0;
        for (int i = 0; i <= in_shape_ - window_; i += stride_) {
            
            // reset stato per considerare finestra indipendente
            if(lstm_ptr) lstm_ptr->resetState();

            //Estraggo finestra
            Matrix<T> x_window = getWindow(X, i);

            //faccio LSTM forward
            Matrix<T> lstm_out = lstm_layer_->Forward(x_window);

            // estraggo ultimo output
            int hidden_size = lstm_out.cols();
            Matrix<T> last_step(1, hidden_size);
            
            const T* lstm_data = lstm_out.Flatten();
            T* last_step_data = last_step.Flatten();
            
            int start_last_row = (window_ - 1) * hidden_size;
            
            for(int k=0; k<hidden_size; ++k) {
                last_step_data[k] = lstm_data[start_last_row + k];
            }

            // faccio dense forward
            Matrix<T> dense_out = dense_layer_->Forward(last_step);

            // infine salvo il risultato
            const T* d_ptr = dense_out.Flatten();
            for(int k=0; k<out_shape_; ++k) {
                out_ptr[window_idx * out_shape_ + k] = d_ptr[k];
            }
            
            window_idx++;
        }

        return latent_output;
    }

    }

#endif