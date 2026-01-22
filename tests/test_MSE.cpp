#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <iomanip>
#include "Matrix.hpp"
#include "solvers/CPUSolver.hpp" 
#include "Loss/MSE.hpp" 

// Helper per confrontare float/double
bool isClose(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

int main() {
    std::cout << " Test MSE Loss" << std::endl;

    using Scalar = double;


    auto solver = std::make_shared<CPUSolver<Scalar>>();
    MSE<Scalar> criterion(solver);

    // Test 1: predizione perfetta
    std::cout << " TEST 1: Perfect Prediction..." << std::endl;
    
    Matrix<Scalar> p1(1, 3);
    p1.Set(0, 0, 1.0); p1.Set(0, 1, -5.0); p1.Set(0, 2, 0.0);
    
    Matrix<Scalar> t1(1, 3);
    t1.Set(0, 0, 1.0); t1.Set(0, 1, -5.0); t1.Set(0, 2, 0.0);

    Matrix<Scalar> res1 = criterion.Compute(p1, t1);
    Scalar val1 = res1.Get(0, 0);

    std::cout << "Loss Calcolata: " << val1 << std::endl;
    if (isClose(val1, 0.0)) {
        std::cout << " PASSED" << std::endl;
    } else {
        std::cerr << " FAILED " << std::endl;
        return 1;
    }

    // Gradient check (dovrebbe uscire 0 se è giusto)
    Matrix<Scalar> g1 = criterion.Gradient();
    if (isClose(g1.Get(0,0), 0.0) && isClose(g1.Get(0,1), 0.0)) {
        std::cout << " PASSED" << std::endl;
    } else {
        std::cerr << " FAILED" << std::endl;
        return 1;
    }


    // Test 2: Calcolo Manuale
    
    std::cout << " TEST 2: Math Verification..." << std::endl;

    

    Matrix<Scalar> p2(1, 2);
    p2.Set(0, 0, 10.0); p2.Set(0, 1, 5.0);

    Matrix<Scalar> t2(1, 2);
    t2.Set(0, 0, 6.0); t2.Set(0, 1, 8.0);

    // Forward
    Matrix<Scalar> res2 = criterion.Compute(p2, t2);
    Scalar val2 = res2.Get(0, 0);
    Scalar expectedLoss = 12.5;

    std::cout << "Loss Attesa: " << expectedLoss << " | Calcolata: " << val2 << std::endl;

    if (isClose(val2, expectedLoss)) {
        std::cout << " PASSED " << std::endl;
    } else {
        std::cerr << " FAILED " << std::endl;
        return 1;
    }
    
    // Backward
    Matrix<Scalar> g2 = criterion.Gradient();
    Scalar grad0 = g2.Get(0, 0);
    Scalar grad1 = g2.Get(0, 1);

    std::cout << "Grad[0] Atteso: 4.0 | Calcolato: " << grad0 << std::endl;
    std::cout << "Grad[1] Atteso: -3.0 | Calcolato: " << grad1 << std::endl;

    if (isClose(grad0, 4.0) && isClose(grad1, -3.0)) {
        std::cout << " PASSED " << std::endl;
    } else {
        std::cerr << " FAILED " << std::endl;
        return 1;
    }

   
}