#include <iostream>
#include <string>
#include <vector>
#include "BenchmarkSuite.hpp"
// Assicurati di includere la tua Factory
// #include "include/SolverFactory.hpp" 

// --- FUNZIONE HELPER TEMPLATE ---
// Il cuore del sistema: viene istanziata per float o double a seconda della scelta runtime
template <typename T>
void run_benchmark_workflow(SolverType type, int max_size) {
    BenchmarkSuite<T> bench;
    
    // Esegue il test di scalabilità che stampa il CSV
    // Assicurati che dentro BenchmarkSuite.hpp la funzione runScalabilityTest
    // stampi SOLO le righe CSV (Size,Time,...) e niente altro testo di debug.
    bench.runScalabilityTest(max_size, type);
}

int main(int argc, char* argv[]) {
    // 1. Controllo Argomenti
    // Ci aspettiamo: ./benchmark_test <solver> <precision>
    if (argc < 3) {
        std::cerr << "ERRORE: Argomenti insufficienti.\n";
        std::cerr << "Uso: " << argv[0] << " <solver_type> <precision>\n";
        std::cerr << "Esempi:\n";
        std::cerr << "  " << argv[0] << " naive float\n";
        std::cerr << "  " << argv[0] << " simd double\n";
        return 1;
    }

    std::string arg_solver = argv[1];
    std::string arg_precision = argv[2];

    // 2. Decodifica Solver (Stringa -> Enum)
    SolverType type;
    if (arg_solver == "naive") {
        type = SolverType::NAIVE;
    } 
    else if (arg_solver == "simd") {
        type = SolverType::SIMD;
    }
    else if (arg_solver == "omp") {
        type = SolverType::OPENMP;
    }
    else {
        std::cerr << "Solver non riconosciuto: " << arg_solver << "\n";
        return 1;
    }

    // 3. Decodifica Precisione e Avvio (Stringa -> Template)
    if (arg_precision == "float") {
        // Chiama la versione float
        run_benchmark_workflow<float>(type, 2048);
    } 
    else if (arg_precision == "double") {
        // Chiama la versione double
        run_benchmark_workflow<double>(type, 2048);
    } 
    else {
        std::cerr << "Precisione non valida: " << arg_precision << " (usa 'float' o 'double')\n";
        return 1;
    }

    return 0;
}
