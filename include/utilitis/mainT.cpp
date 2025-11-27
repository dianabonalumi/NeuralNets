#include "BenchmarkSuite.hpp"


// Funzione principale di esecuzione dei benchmark sulla velocità di moltiplicazione delle matrici

int main() {
    // Istanzio la suite per i float (o double)
    BenchmarkSuite<double> bench;

    std::cout << "=== INIZIO BENCHMARK MATRICI ===\n";

    // Test 1: Piccola (per debug)
    bench.runTest(64, 64, 64, SolverType::NAIVE);

    // Test 2: Media (per vedere se l'unrolling aiuta)
    bench.runTest(256, 256, 256, SolverType::NAIVE);
    // bench.runTest(256, 256, 256, SolverType::SIMD);

    // Test 3: Grande (per vedere il Cache Blocking brillare)
    // Nota: Il Naive qui ci metterà un'eternità
    bench.runTest(1024, 1024, 1024, SolverType::NAIVE); 

    bench.runScalabilityTest(2048, SolverType::NAIVE);

    return 0;
}
