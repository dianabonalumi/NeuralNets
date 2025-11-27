#include <iostream>
#include <string>
#include <vector>
#include "BenchmarkSuite.hpp"

template <typename T>
void run_benchmark_workflow(SolverType type, int max_size) {
    BenchmarkSuite<T> bench;
    

    bench.runScalabilityTest(max_size, type);
}

int main(int argc, char* argv[]) {

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
    else if (arg_solver == "unroll") {
        type = SolverType::UNROLL;
    }
    else if (arg_solver == "tiling") {
        type = SolverType::TILING;
    }
    else if (arg_solver == "simd_unrolling") {
        type = SolverType::SIMD_UNROLL_1D;
    }
    else if (arg_solver == "simd_unroll_2d") {
        type = SolverType::SIMD_UNROLL_2D;
    }
    else {
        std::cerr << "Solver non riconosciuto: " << arg_solver << "\n";
        return 1;
    }

    if (arg_precision == "float") {
        run_benchmark_workflow<float>(type, 2048);
    } 
    else if (arg_precision == "double") {
        run_benchmark_workflow<double>(type, 2048);
    } 
    else {
        std::cerr << "Precisione non valida: " << arg_precision << " (usa 'float' o 'double')\n";
        return 1;
    }

    return 0;
}
