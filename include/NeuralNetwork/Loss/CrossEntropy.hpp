#ifndef CROSS_ENTROPY_HPP
#define CROSS_ENTROPY_HPP

#include "Loss.hpp"
#include <cmath>        // Per std::log
#include <stdexcept>    // Per eccezioni
#include <limits>       // Per epsilon se necessario

template <typename T>
class CrossEntropy : public Loss<T> {
private:
    Matrix<T> lastX; // Salva le prediction per il calcolo del gradiente
    Matrix<T> lastY; // Salva i target per il calcolo del gradiente
    
    // definisco una costante epsilon per evitare di avere log0
    const T epsilon = static_cast<T>(1e-9);

public:
    // Costruttore
    CrossEntropy(std::shared_ptr<Matrix_Solver<T>> solver) : Loss<T>(solver) {}

    // -------------------------------------------------------------------------
    // COMPUTE: Calcola la Loss
    // Formula: L = - (1/N) * sum( target * log(prediction + epsilon) )
    // -------------------------------------------------------------------------
    Matrix<T> Compute(const Matrix<T> prediction, const Matrix<T> target) override {
        //check delle dimensioni
        if (prediction.rows() != target.rows() || prediction.cols() != target.cols()) {
            throw std::invalid_argument("CrossEntropy Error: Dimensioni di Prediction e Target non corrispondono.");
        }

    
        this->lastX = prediction;
        this->lastY = target;

        size_t rows = prediction.rows();
        size_t cols = prediction.cols();
        size_t total_elements = rows * cols;

        //aggiunta per migliorare la cache-friendly :)
        const T* predData = prediction.Flatten();
        const T* targData = target.Flatten();

        T total_loss = 0;

        // Calcolo di - sum(target * log(prediction))
        for (size_t i = 0; i < total_elements; ++i) {
            T p = predData[i];
            T t = targData[i];

            
            T safe_p = p + epsilon; 
            
            // Cross Entropy formula
            total_loss += -t * std::log(safe_p);
        }

        // 4. Calcolo della media sulla Batch (divisione per numero di righe)
        // Nota: Di solito si normalizza per il numero di campioni (rows), non per tutti gli elementi.
        T mean_loss = total_loss / static_cast<T>(rows);

        // il risultato è uno scalare
        Matrix<T> result(1, 1);
        result.Set(0, 0, mean_loss);
        
        return result;
    }

}