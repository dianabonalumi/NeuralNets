#include <iostream>
#include <memory>
#include <cassert>
#include <cmath>
#include <vector>

#include "../include/NeuralNetwork/Layer/LSTM.hpp"
#include "../include/NeuralNetwork/Layer/Layer.hpp"
#include "../include/NeuralNetwork/Matrix.hpp"
#include "../include/NeuralNetwork/Optimizer/Adam.hpp"
#include "../include/factory_m.hpp"

// Helper to convert unique_ptr to shared_ptr for solver
std::shared_ptr<Matrix_Solver<float>> getSolver() {
    auto unique_solver = SolverFactory<float>::createSolver(SolverType::ALL);
    // Convert unique_ptr to shared_ptr
    std::shared_ptr<Matrix_Solver<float>> solver(unique_solver.release());
    return solver;
}

// Test utility functions
bool isClose(float a, float b, float tol = 1e-5f) {
    return std::abs(a - b) < tol;
}

bool matrixClose(const Matrix<float>& a, const Matrix<float>& b, float tol = 1e-5f) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }
    const float* a_data = a.Flatten();
    const float* b_data = b.Flatten();
    size_t total = a.rows() * a.cols();
    for (size_t i = 0; i < total; ++i) {
        if (!isClose(a_data[i], b_data[i], tol)) {
            return false;
        }
    }
    return true;
}

void printMatrix(const Matrix<float>& m, const std::string& name = "") {
    if (!name.empty()) std::cout << name << " ";
    std::cout << "(" << m.rows() << "x" << m.cols() << "):" << std::endl;
    int max_rows = (int)std::min((size_t)5, m.rows());
    int max_cols = (int)std::min((size_t)5, m.cols());
    for (int i = 0; i < max_rows; i++) {
        for (int j = 0; j < max_cols; j++) {
            std::cout << m.Get(i, j) << "\t";
        }
        if ((size_t)max_cols < m.cols()) std::cout << "...";
        std::cout << std::endl;
    }
    if ((size_t)max_rows < m.rows()) std::cout << "...\n";
    std::cout << std::endl;
}

// Test 1: LSTM instantiation and weight initialization
void test_lstm_instantiation() {
    std::cout << "TEST 1: LSTM Instantiation and Weight Initialization" << std::endl;
    
    auto solver = getSolver();
    auto optimizer = std::make_shared<Adam<float>>(solver, 0.001f);
    
    int input_features = 10;
    int hidden_size = 20;
    
    LSTM<float> lstm(solver, optimizer, input_features, hidden_size);
    lstm.WeightInitialization(WeightInit::Xavier);
    
    std::cout << "✓ LSTM created successfully with input_features=" << input_features 
              << ", hidden_size=" << hidden_size << std::endl;
}

// Test 2: LSTM forward pass with single batch (BPTT: store history)
void test_lstm_forward_single() {
    std::cout << "\nTEST 2: LSTM Forward Pass - Single Timestep" << std::endl;
    
    auto solver = getSolver();
    auto optimizer = std::make_shared<Adam<float>>(solver, 0.001f);
    
    int batch_size = 2;
    int input_features = 5;
    int hidden_size = 8;
    
    LSTM<float> lstm(solver, optimizer, input_features, hidden_size);
    lstm.WeightInitialization(WeightInit::Xavier);
    
    // Create input: batch_size x input_features
    Matrix<float> input(batch_size, input_features);
    float* input_data = input.Flatten();
    for (int i = 0; i < batch_size * input_features; ++i) {
        input_data[i] = static_cast<float>(i) * 0.1f;
    }
    
    printMatrix(input, "Input");
    
    // Forward pass (adds to history)
    Matrix<float> output = lstm.Forward(input);
    
    // Check output dimensions
    assert(output.rows() == batch_size);
    assert(output.cols() == hidden_size);
    
    std::cout << "✓ Output dimensions correct: " << output.rows() << "x" << output.cols() << std::endl;
    printMatrix(output, "Output");
}

// Test 3: LSTM sequence processing with BPTT (multiple forwards, single backward)
void test_lstm_bptt_sequence() {
    std::cout << "\nTEST 3: LSTM BPTT - Sequence Processing" << std::endl;
    
    auto solver = getSolver();
    auto optimizer = std::make_shared<Adam<float>>(solver, 0.001f);
    
    int batch_size = 2;
    int input_features = 4;
    int hidden_size = 6;
    int sequence_length = 3;
    
    LSTM<float> lstm(solver, optimizer, input_features, hidden_size);
    lstm.WeightInitialization(WeightInit::Xavier);
    
    std::vector<Matrix<float>> outputs;
    
    // Forward phase: process entire sequence, storing history
    for (int t = 0; t < sequence_length; ++t) {
        Matrix<float> input(batch_size, input_features);
        float* input_data = input.Flatten();
        for (int i = 0; i < batch_size * input_features; ++i) {
            input_data[i] = static_cast<float>(t * input_features + i) * 0.05f;
        }
        
        Matrix<float> output = lstm.Forward(input);
        outputs.push_back(output);
        
        std::cout << "  Timestep " << t << " output size: " << output.rows() << "x" << output.cols() << std::endl;
    }
    
    assert(outputs.size() == sequence_length);
    std::cout << "✓ Forward pass through sequence successful with " << sequence_length << " timesteps" << std::endl;
    
    // Backward phase: single backward call processes entire sequence via stored history
    Matrix<float> grad_last = outputs.back(); // Use last hidden state as initial gradient
    float* grad_data = grad_last.Flatten();
    for (int i = 0; i < batch_size * hidden_size; ++i) {
        grad_data[i] = 0.05f;
    }
    
    Matrix<float> grad_input = lstm.Backward(grad_last);
    
    assert(grad_input.rows() == batch_size);
    assert(grad_input.cols() == input_features);
    
    std::cout << "✓ Backward pass through entire sequence successful" << std::endl;
    std::cout << "✓ Gradient dimensions match input: " << grad_input.rows() << "x" << grad_input.cols() << std::endl;
}

// Test 4: LSTM sequence processing (old test replaced - see test_lstm_bptt_sequence)
void test_lstm_long_sequence() {
    std::cout << "\nTEST 4: LSTM Long Sequence BPTT" << std::endl;
    
    auto solver = getSolver();
    auto optimizer = std::make_shared<Adam<float>>(solver, 0.001f);
    
    int batch_size = 2;
    int input_features = 4;
    int hidden_size = 6;
    int sequence_length = 5;
    
    LSTM<float> lstm(solver, optimizer, input_features, hidden_size);
    lstm.WeightInitialization(WeightInit::Xavier);
    
    // Process longer sequence
    Matrix<float> last_output;
    for (int t = 0; t < sequence_length; ++t) {
        Matrix<float> input(batch_size, input_features);
        float* input_data = input.Flatten();
        for (int i = 0; i < batch_size * input_features; ++i) {
            input_data[i] = static_cast<float>(t % 3) * 0.05f; // Cyclical input
        }
        
        last_output = lstm.Forward(input);
    }
    
    // Single backward call processes entire stored sequence
    Matrix<float> grad(batch_size, hidden_size);
    float* grad_data = grad.Flatten();
    for (int i = 0; i < batch_size * hidden_size; ++i) {
        grad_data[i] = 0.01f;
    }
    
    Matrix<float> grad_input = lstm.Backward(grad);
    
    assert(grad_input.rows() == batch_size);
    assert(grad_input.cols() == input_features);
    
    std::cout << "✓ Long sequence (" << sequence_length << " steps) processed successfully" << std::endl;
}

// Test 5: LSTM state reset between sequences
void test_lstm_state_reset() {
    std::cout << "\nTEST 5: LSTM State Reset Between Sequences" << std::endl;
    
    auto solver = getSolver();
    auto optimizer = std::make_shared<Adam<float>>(solver, 0.001f);
    
    int batch_size = 2;
    int input_features = 4;
    int hidden_size = 6;
    
    LSTM<float> lstm(solver, optimizer, input_features, hidden_size);
    lstm.WeightInitialization(WeightInit::Xavier);
    
    // Sequence 1: forward and backward
    Matrix<float> input1(batch_size, input_features);
    float* data = input1.Flatten();
    for (int i = 0; i < batch_size * input_features; ++i) {
        data[i] = 0.1f;
    }
    Matrix<float> output1 = lstm.Forward(input1);
    
    Matrix<float> grad1(batch_size, hidden_size);
    float* gdata = grad1.Flatten();
    for (int i = 0; i < batch_size * hidden_size; ++i) {
        gdata[i] = 0.01f;
    }
    lstm.Backward(grad1);
    
    std::cout << "✓ Sequence 1 processed" << std::endl;
    
    // Reset state
    lstm.resetState();
    std::cout << "✓ State reset called" << std::endl;
    
    // Sequence 2: same input should give similar structure (but different due to weight updates)
    Matrix<float> input2(batch_size, input_features);
    data = input2.Flatten();
    for (int i = 0; i < batch_size * input_features; ++i) {
        data[i] = 0.1f;
    }
    Matrix<float> output2 = lstm.Forward(input2);
    
    assert(output2.rows() == output1.rows());
    assert(output2.cols() == output1.cols());
    std::cout << "✓ Output dimensions consistent after reset" << std::endl;
}

// Test 6: LSTM with different batch sizes
void test_lstm_batch_sizes() {
    std::cout << "\nTEST 6: LSTM with Different Batch Sizes" << std::endl;
    
    auto solver = getSolver();
    auto optimizer = std::make_shared<Adam<float>>(solver, 0.001f);
    
    int input_features = 5;
    int hidden_size = 8;
    
    LSTM<float> lstm(solver, optimizer, input_features, hidden_size);
    lstm.WeightInitialization(WeightInit::Xavier);
    
    std::vector<int> batch_sizes = {1, 2, 4, 8};
    
    for (int batch_size : batch_sizes) {
        lstm.resetState();
        
        Matrix<float> input(batch_size, input_features);
        float* data = input.Flatten();
        for (int i = 0; i < batch_size * input_features; ++i) {
            data[i] = 0.1f;
        }
        
        Matrix<float> output = lstm.Forward(input);
        
        assert(output.rows() == batch_size);
        assert(output.cols() == hidden_size);
        
        std::cout << "  ✓ Batch size " << batch_size << " processed successfully" << std::endl;
    }
    
    std::cout << "✓ All batch sizes handled correctly" << std::endl;
}

// Test 7: Gradient flow check (numerical stability)
void test_lstm_gradient_flow() {
    std::cout << "\nTEST 7: LSTM Gradient Flow and Numerical Stability" << std::endl;
    
    auto solver = getSolver();
    auto optimizer = std::make_shared<Adam<float>>(solver, 0.001f);
    
    int batch_size = 2;
    int input_features = 4;
    int hidden_size = 6;
    
    LSTM<float> lstm(solver, optimizer, input_features, hidden_size);
    lstm.WeightInitialization(WeightInit::Xavier);
    
    // Create input
    Matrix<float> input(batch_size, input_features);
    float* input_data = input.Flatten();
    for (int i = 0; i < batch_size * input_features; ++i) {
        input_data[i] = static_cast<float>(i) * 0.01f;
    }
    
    // Forward
    Matrix<float> output = lstm.Forward(input);
    
    // Check output values are reasonable (not NaN or Inf)
    const float* output_data = output.Flatten();
    for (int i = 0; i < output.rows() * output.cols(); ++i) {
        assert(!std::isnan(output_data[i]) && !std::isinf(output_data[i]));
    }
    std::cout << "✓ Forward pass produces valid values (no NaN/Inf)" << std::endl;
    
    // Create gradient
    Matrix<float> grad(batch_size, hidden_size);
    float* grad_data = grad.Flatten();
    for (int i = 0; i < batch_size * hidden_size; ++i) {
        grad_data[i] = 0.01f;
    }
    
    // Backward
    Matrix<float> grad_input = lstm.Backward(grad);
    
    // Check gradient values
    const float* grad_input_data = grad_input.Flatten();
    for (int i = 0; i < grad_input.rows() * grad_input.cols(); ++i) {
        assert(!std::isnan(grad_input_data[i]) && !std::isinf(grad_input_data[i]));
    }
    std::cout << "✓ Backward pass produces valid gradients (no NaN/Inf)" << std::endl;
}

// Test 8: Training stability check - multiple iterations
void test_lstm_training_stability() {
    std::cout << "\nTEST 8: LSTM Training Stability (Multiple Iterations)" << std::endl;
    
    auto solver = getSolver();
    auto optimizer = std::make_shared<Adam<float>>(solver, 0.001f);
    
    int batch_size = 4;
    int input_features = 5;
    int hidden_size = 8;
    int sequence_length = 3;
    int num_iterations = 5;
    
    LSTM<float> lstm(solver, optimizer, input_features, hidden_size);
    lstm.WeightInitialization(WeightInit::Xavier);
    
    float prev_grad_norm = 0;
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        lstm.resetState();
        
        // Process sequence
        Matrix<float> last_output;
        for (int t = 0; t < sequence_length; ++t) {
            Matrix<float> input(batch_size, input_features);
            float* input_data = input.Flatten();
            for (int i = 0; i < batch_size * input_features; ++i) {
                input_data[i] = std::sin(static_cast<float>(iter + t + i) * 0.5f) * 0.5f;
            }
            last_output = lstm.Forward(input);
        }
        
        // Backward
        Matrix<float> grad(batch_size, hidden_size);
        float* grad_data = grad.Flatten();
        float grad_norm = 0;
        for (int i = 0; i < batch_size * hidden_size; ++i) {
            grad_data[i] = 0.01f;
            grad_norm += grad_data[i] * grad_data[i];
        }
        grad_norm = std::sqrt(grad_norm);
        
        Matrix<float> grad_input = lstm.Backward(grad);
        
        // Check for explosion
        const float* g_data = grad_input.Flatten();
        float max_grad = 0;
        for (int i = 0; i < grad_input.rows() * grad_input.cols(); ++i) {
            assert(!std::isnan(g_data[i]) && !std::isinf(g_data[i]));
            max_grad = std::max(max_grad, std::abs(g_data[i]));
        }
        
        assert(max_grad < 1000.0f); // Gradient shouldn't explode
        
        std::cout << "  Iteration " << iter << ": max grad = " << max_grad << std::endl;
    }
    
    std::cout << "✓ Training stability check passed - no gradient explosion over " << num_iterations << " iterations" << std::endl;
}

int main() {
    std::cout << "======================================" << std::endl;
    std::cout << "LSTM Unit Tests" << std::endl;
    std::cout << "======================================" << std::endl;
    
    try {
        test_lstm_instantiation();
        test_lstm_forward_single();
        test_lstm_bptt_sequence();
        test_lstm_long_sequence();
        test_lstm_state_reset();
        test_lstm_batch_sizes();
        test_lstm_gradient_flow();
        test_lstm_training_stability();
        
        std::cout << "\n======================================" << std::endl;
        std::cout << "ALL TESTS PASSED ✓" << std::endl;
        std::cout << "======================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
