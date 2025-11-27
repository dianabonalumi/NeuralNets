#ifndef BENCHMARK_SUITE_HPP
#define BENCHMARK_SUITE_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cblas.h> // Header di OpenBLAS
#include "../factory_m.hpp" // La tua Factory

// --- TRUCCO PER OPENBLAS ---
// Creiamo due funzioni wrapper per gestire la differenza tra float e double
// OpenBLAS usa nomi diversi (cblas_sgemm per float, cblas_dgemm per double)
template <typename T>
void call_openblas(int M, int N, int K, const T* A, const T* B, T* C);

// Specializzazione per FLOAT
template <>
void call_openblas<float>(int M, int N, int K, const float* A, const float* B, float* C) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

// Specializzazione per DOUBLE
template <>
void call_openblas<double>(int M, int N, int K, const double* A, const double* B, double* C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 1.0, A, K, B, N, 0.0, C, N);
}
// ---------------------------

template <typename T>
class BenchmarkSuite {
private:
    // Helper per inizializzare dati random
    void randomInit(std::vector<T>& vec) {
        for (auto& v : vec) {
            v = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }
    }

    // Helper per verificare se due matrici sono uguali (con tolleranza)
    bool verify(int size, const T* expected, const T* actual) {
        double epsilon = 1e-2; // Tolleranza
        for (int i = 0; i < size; ++i) {
            if (std::abs(expected[i] - actual[i]) > epsilon) {
                std::cerr << "ERRORE all'indice " << i 
                          << ": Atteso " << expected[i] 
                          << ", Ottenuto " << actual[i] << "\n";
                return false;
            }
        }
        return true;
    }

public:
    // Il metodo principale che lanci dal main
    void runTest(int M, int N, int K, SolverType type) {
        // 1. Setup Dati
        std::vector<T> A(M * K);
        std::vector<T> B(K * N);
        std::vector<T> C_mine(M * N, 0); // Output del tuo codice
        std::vector<T> C_blas(M * N, 0); // Output di OpenBLAS (Corretto)

        randomInit(A);
        randomInit(B);

        // 2. Esecuzione OPENBLAS (Il riferimento)
        auto start_blas = std::chrono::high_resolution_clock::now();
        call_openblas<T>(M, N, K, A.data(), B.data(), C_blas.data());
        auto end_blas = std::chrono::high_resolution_clock::now();
        double time_blas = std::chrono::duration<double, std::milli>(end_blas - start_blas).count();

        // 3. Esecuzione TUO SOLVER
        auto solver = SolverFactory<T>::createSolver(type);
        std::cout << "Testando: " << std::setw(15) << solver->getName() 
                  << " | Matrice " << M << "x" << N << "x" << K;

        auto start_mine = std::chrono::high_resolution_clock::now();
        solver->multiply(M, N, K, A.data(), B.data(), C_mine.data());
        auto end_mine = std::chrono::high_resolution_clock::now();
        double time_mine = std::chrono::duration<double, std::milli>(end_mine - start_mine).count();

        // 4. Confronto e Report
        bool passed = verify(M * N, C_blas.data(), C_mine.data());

        if (passed) {
            double speedup = time_mine / time_blas; // Quante volte siamo più lenti di BLAS
            std::cout << " | STATUS: OK"
                      << " | Tuo Tempo: " << std::fixed << std::setprecision(2) << time_mine << "ms"
                      << " | BLAS Tempo: " << time_blas << "ms"
                      << " | Gap: " << speedup << "x slower" << std::endl;
        } else {
            std::cout << " | STATUS: FALLITO (Risultati diversi!)" << std::endl;
        }
    }

    // In BenchmarkSuite.hpp

void runScalabilityTest(int max_n, SolverType type) {
    // 1. HEADER CSV AGGIORNATO: Aggiungiamo le colonne per BLAS
    std::cout << "Size,Time_Mine,GFLOPs_Mine,Time_Blas,GFLOPs_Blas\n"; 

    // Incremento lineare (es. +64 o +128) per avere grafici fluidi
    for (int n = 128; n <= max_n; n += 128) {
        
        // Setup Dati
        std::vector<T> A(n * n);
        std::vector<T> B(n * n);
        std::vector<T> C_mine(n * n);
        std::vector<T> C_blas(n * n);
        randomInit(A); randomInit(B);

        // --- MISURA IL TUO SOLVER ---
        auto solver = SolverFactory<T>::createSolver(type);
        auto start_mine = std::chrono::high_resolution_clock::now();
        solver->multiply(n, n, n, A.data(), B.data(), C_mine.data());
        auto end_mine = std::chrono::high_resolution_clock::now();
        
        double duration_mine = std::chrono::duration<double>(end_mine - start_mine).count();
        double ms_mine = duration_mine * 1000.0;
        double gflops_mine = (2.0 * std::pow(n, 3)) / (duration_mine * 1e9);

        // --- MISURA OPENBLAS (BASELINE) ---
        auto start_blas = std::chrono::high_resolution_clock::now();
        call_openblas<T>(n, n, n, A.data(), B.data(), C_blas.data());
        auto end_blas = std::chrono::high_resolution_clock::now();

        double duration_blas = std::chrono::duration<double>(end_blas - start_blas).count();
        double ms_blas = duration_blas * 1000.0;
        double gflops_blas = (2.0 * std::pow(n, 3)) / (duration_blas * 1e9);

        // --- STAMPA CSV COMPLETO ---
        // Formato: Size, TuoTempo, TuoiGflops, BlasTempo, BlasGflops
        std::cout << n << "," 
                  << ms_mine << "," << gflops_mine << ","
                  << ms_blas << "," << gflops_blas << "\n";
    }
}

};





#endif
