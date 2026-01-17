#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include "Architecture.hpp"

#include "../Loss/Loss.hpp"

#include <string>
#include <fstream>

#include "Encoder.hpp"
#include "Decoder.hpp"

template<typename T>
class Autoencoder: public Architecture<T> {
private:
    Encoder<T> enc;
    Decoder<T> dec;
    const std::shared_ptr<Loss<T>> loss_;

    Matrix<T> pred, grad;

public:
    Autoencoder<T>(const std::shared_ptr<Matrix_Solver<T>>& solver,
        const std::shared_ptr<Optimizer<T>>& optimizer,
        const std::shared_ptr<Loss<T>>& loss,
        const int in_shape, const int bottleneck_shape, const int hidden_state,
        const int window_size, const int stride): loss_(loss), 
        enc(solver, optimizer, window_size, stride, in_shape, bottleneck_shape, hidden_state),
        dec(solver, optimizer, bottleneck_shape, in_shape) {
        }

    Matrix<T> Predict(const Matrix<T>& X) {
        Matrix<T> encoded = this->enc.Predict(X);

        this->pred = this->dec.Predict(encoded);

        return this->pred;
    }

    Matrix<T> Eval(const Matrix<T>& target) {
        Matrix<T> result = this->loss_->Compute(this->pred, target);

        this->grad = this->loss_->Gradient();

        return result;
    }

    Matrix<T> Backward() {
        this->dec.SetGradient(this->grad);

        Matrix<T> grad = this->dec.Backward();

        this->enc.SetGradient(grad);
        grad = this->enc.Backward();

        return grad;
    }

    Encoder<T> GetEncoder() {
        return this->enc;
    }

    Decoder<T> GetDecoder() {
        return this->dec;
    }

    void Save(std::string pos) {
        // Serialize both encoder and decoder
        T* enc_data = enc.Serialize();
        T* dec_data = dec.Serialize();
        
        size_t enc_size = static_cast<size_t>(enc_data[0]) + 1;
        size_t dec_size = static_cast<size_t>(dec_data[0]) + 1;
        
        // Write encoder data
        std::ofstream enc_file(pos + "_enc.bin", std::ios::binary);
        enc_file.write(reinterpret_cast<char*>(enc_data), static_cast<std::streamsize>(enc_size * sizeof(T)));
        enc_file.close();
        
        // Write decoder data
        std::ofstream dec_file(pos + "_dec.bin", std::ios::binary);
        dec_file.write(reinterpret_cast<char*>(dec_data), static_cast<std::streamsize>(dec_size * sizeof(T)));
        dec_file.close();
        
        delete[] enc_data;
        delete[] dec_data;
    }

    void Restore(std::string pos) {
        // Read encoder data
        std::ifstream enc_file(pos + "_enc.bin", std::ios::binary);
        if (!enc_file.is_open()) return;
        
        enc_file.seekg(0, std::ios::end);
        std::streamsize enc_bytes = enc_file.tellg();
        enc_file.seekg(0, std::ios::beg);
        
        size_t enc_size = enc_bytes / sizeof(T);
        T* enc_data = new T[enc_size];
        enc_file.read(reinterpret_cast<char*>(enc_data), enc_bytes);
        enc_file.close();
        
        // Read decoder data
        std::ifstream dec_file(pos + "_dec.bin", std::ios::binary);
        if (!dec_file.is_open()) {
            delete[] enc_data;
            return;
        }
        
        dec_file.seekg(0, std::ios::end);
        std::streamsize dec_bytes = dec_file.tellg();
        dec_file.seekg(0, std::ios::beg);
        
        size_t dec_size = dec_bytes / sizeof(T);
        T* dec_data = new T[dec_size];
        dec_file.read(reinterpret_cast<char*>(dec_data), dec_bytes);
        dec_file.close();
        
        // Deserialize both
        enc.Deserialize(enc_data);
        dec.Deserialize(dec_data);
        
        delete[] enc_data;
        delete[] dec_data;
    }
};

#endif