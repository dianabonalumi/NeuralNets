#include <iostream>
#include <vector>
#include <iomanip> // Per std::setw (formattazione output)
#include <chrono>  // Per misurare il tempo (primo test)

// Assicurati che il percorso sia corretto rispetto a dove compili
#include "../include/factory_m.hpp"

// Funzione helper per stampare le matrici in modo leggibile
template <typename T>
void printMatrix(const std::string& name, int rows, int cols, const T* data) {
    std::cout << "Matrice " << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // data[i * cols + j] è la formula per accedere all'array 1D
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                      << data[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "=== Test Architettura Matrix Solver ===\n\n";

    // 1. DEFINIZIONE DATI
    // Usiamo dimensioni piccole per verificare i calcoli a mano
    // Calcoliamo: C = A x B
    // A è 2x3, B è 3x2 -> C sarà 2x2
    int M = 2; 
    int K = 3; 
    int N = 2;

    // Usiamo std::vector per gestire la memoria nel main in modo sicuro
    // Ma passeremo i puntatori grezzi (.data()) al solver
    std::vector<float> A = {
        1.0f, 2.0f, 3.0f,  // Riga 0
        4.0f, 5.0f, 6.0f   // Riga 1
    };

    std::vector<float> B = {
        1.0f, 0.0f,  // Riga 0
        0.0f, 1.0f,  // Riga 1
        1.0f, 1.0f   // Riga 2
    };

    // C deve essere grande abbastanza (MxN)
    std::vector<float> C(M * N, 0.0f);

    // Stampa Input
    printMatrix("A", M, K, A.data());
    printMatrix("B", K, N, B.data());

    // 2. CREAZIONE DEL SOLVER (FACTORY)
    // Qui chiediamo esplicitamente il NAIVE, oppure non passiamo nulla (default)
    std::cout << "-> Creazione del Solver tramite Factory...\n";
    auto solver = SolverFactory<float>::createSolver(SolverType::UNROLL);

    if (!solver) {
        std::cerr << "Errore: Impossibile creare il solver!\n";
        return -1;
    }
    

    // 3. ESECUZIONE (con timer semplice)
    std::cout << "-> Esecuzione moltiplicazione...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    // Passiamo i puntatori raw (.data()) come richiede l'interfaccia
    solver->multiply(M, N, K, A.data(), B.data(), C.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "-> Tempo impiegato: " << elapsed.count() << " ms\n\n";

    // 4. VERIFICA RISULTATO
    // Calcolo atteso:
    // C[0][0] = 1*1 + 2*0 + 3*1 = 4
    // C[0][1] = 1*0 + 2*1 + 3*1 = 5
    // C[1][0] = 4*1 + 5*0 + 6*1 = 10
    // C[1][1] = 4*0 + 5*1 + 6*1 = 11
    printMatrix("C (Risultato)", M, N, C.data());

    std::cout << "Se vedi [4, 5, 10, 11] sopra, il codice funziona!\n";

    return 0;
}
