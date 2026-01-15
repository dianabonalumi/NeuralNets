#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>
#include <cassert>
#include "Matrix.hpp"
#include "solvers/CPUSolver.hpp" 
#include "Loss/CrossEntropy.hpp" 

bool isClose(double a, double b, double tol = 1e-5) {
    return std::abs(a - b) < tol;
}

int main() {


    using Scalar = double;


    auto solver = std::make_shared<CPUSolver<Scalar>>();
    CrossEntropy<Scalar> criterion(solver);

    //1. test sul calcolo numerico
    std::cout << " TEST 1: Verifica Compute (Forward)..." << std::endl;

    // test su una batch di 2 esempi e 3 classi
    size_t batch_size = 2;
    size_t n_classes = 3;

    Matrix<Scalar> pred(batch_size, n_classes);
    Matrix<Scalar> target(batch_size, n_classes);

    // Esempio 1
    pred.Set(0, 0, 0.1); pred.Set(0, 1, 0.8); pred.Set(0, 2, 0.1);
    target.Set(0, 0, 0.0); target.Set(0, 1, 1.0); target.Set(0, 2, 0.0);

    // Esempio 2
    pred.Set(1, 0, 0.4); pred.Set(1, 1, 0.4); pred.Set(1, 2, 0.2);
    target.Set(1, 0, 1.0); target.Set(1, 1, 0.0); target.Set(1, 2, 0.0);

    // Compute
    Matrix<Scalar> lossMat = criterion.Compute(pred, target);
    Scalar lossValue = lossMat.Get(0, 0);

    
    Scalar expectedLoss = 0.569715;

    std::cout << "Loss Calcolata: " << lossValue << std::endl;
    std::cout << "Loss Attesa:    " << expectedLoss << std::endl;

    if (isClose(lossValue, expectedLoss)) {
        std::cout << " PASSED" << std::endl;
    } else {
        std::cerr << "FAILED" << std::endl;
        return 1;
    }

    
    // TEST 2: Verifica Gradiente

    

    //eseguo gradient
    Matrix<Scalar> grad = criterion.Gradient();

    Scalar g_0_1 = grad.Get(0, 1);
    Scalar expected_g_0_1 = -0.625;

    // Gradiente Esempio 1, Classe 0 (Target=0, Pred=0.1)
  
    Scalar g_0_0 = grad.Get(0, 0);
    Scalar expected_g_0_0 = 0.0;

    // Gradiente Esempio 2, Classe 0 (Target=1, Pred=0.4)
    
    Scalar g_1_0 = grad.Get(1, 0);
    Scalar expected_g_1_0 = -1.25;

    std::cout << "Grad(0,1) [T=1, P=0.8]: Atteso " << expected_g_0_1 << ", Ottenuto " << g_0_1 << std::endl;
    std::cout << "Grad(1,0) [T=1, P=0.4]: Atteso " << expected_g_1_0 << ", Ottenuto " << g_1_0 << std::endl;

    if (isClose(g_0_1, expected_g_0_1) && isClose(g_1_0, expected_g_1_0) && isClose(g_0_0, expected_g_0_0)) {
        std::cout << " PASSED" << std::endl;
    } else {
        std::cerr << " FAILED " << std::endl;
        return 1;
    }

    // Test 3: Verifica stabilità
    
    std::cout << " TEST 3: Verifica Stabilità (Input 0) " << std::endl;

    Matrix<Scalar> predZero(1, 2);
    Matrix<Scalar> targetZero(1, 2);
    
    // Prediciamo 0 esatto dove il target è 1 (errore massimo)
    
    predZero.Set(0, 0, 0.0); predZero.Set(0, 1, 1.0);
    targetZero.Set(0, 0, 1.0); targetZero.Set(0, 1, 0.0); 

    Matrix<Scalar> lossZero = criterion.Compute(predZero, targetZero);
    
    std::cout << "Loss con predizione 0: " << lossZero.Get(0, 0) << std::endl;

    if (!std::isinf(lossZero.Get(0, 0)) && !std::isnan(lossZero.Get(0, 0))) {
        std::cout << " PASSED" << std::endl;
    } else {
        std::cerr << " FAILED" << std::endl;
        return 1;
    }


    // TEST 4: Error Handling Dimensioni
    
    std::cout << " TEST 4: Verifica Eccezione Dimensioni." << std::endl;

    Matrix<Scalar> badTarget(batch_size + 1, n_classes); // Dimensione errata
    bool caught = false;
    try {
        criterion.Compute(pred, badTarget);
    } catch (const std::invalid_argument& e) {
        std::cout << "Eccezione catturata correttamente: " << e.what() << std::endl;
        caught = true;
    }

    if (caught) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cerr << "FAILED" << std::endl;
        return 1;
    }

    

    return 0;
}