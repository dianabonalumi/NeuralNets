#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <stdexcept>
#include <algorithm>
#include <utility> 

template <typename T>
class Matrix {
private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    T* data_ = nullptr;
    // REMOVED: bool owns_data_ = true; (Ownership is always assumed)

    void cleanup() {
        // We always assume ownership of data_, so we always delete[] if allocated.
        if (data_ != nullptr) {
            delete[] data_;
        }
        data_ = nullptr;
        rows_ = 0;
        cols_ = 0;
    }

    size_t getIndex(size_t r, size_t c) const {
        return r * cols_ + c;
    }

    void checkBounds(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix indices out of bounds.");
        }
    }

public:
    // Default Constructor
    Matrix() = default; 

    // Destructor (Rule of Five)
    ~Matrix() {
        cleanup();
    }
    
    // --- Rule of Five Implementation (Deep Copy/Move) ---

    // 1. Copy Constructor (Deep Copy)
    Matrix(const Matrix& other) 
        : rows_(other.rows_), cols_(other.cols_), data_(nullptr) 
    {
        if (other.data_ != nullptr) {
            size_t size = rows_ * cols_;
            data_ = new T[size]; // Allocate NEW memory
            std::copy(other.data_, other.data_ + size, data_); 
        }
    }

    // 2. Copy Assignment Operator (Deep Copy)
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            cleanup(); // Clean up current memory
            
            rows_ = other.rows_;
            cols_ = other.cols_;
            
            if (other.data_ != nullptr) {
                size_t size = rows_ * cols_;
                data_ = new T[size]; // Allocate NEW memory
                std::copy(other.data_, other.data_ + size, data_);
            }
        }
        return *this;
    }

    // 3. Move Constructor (Steal Ownership)
    Matrix(Matrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(other.data_) 
    {
        // Steal resources and leave 'other' in an empty, safe state
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_ = nullptr;
    }

    // 4. Move Assignment Operator (Steal Ownership)
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            cleanup(); // Clean up current memory (optional but safe)

            // Steal resources
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;

            // Leave 'other' in an empty, safe state
            other.rows_ = 0;
            other.cols_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }
    
    // --- Primary Constructor/Accessors ---

    // Constructor: Allocates and owns memory
    Matrix(size_t r, size_t c) : rows_(r), cols_(c) {
        if (r == 0 || c == 0) {
            data_ = nullptr;
            rows_ = 0;
            cols_ = 0;
        } else {
            size_t size = r * c;
            data_ = new T[size] {}; 
        }
    }

    void Set(size_t r, size_t c, const T& val) {
        checkBounds(r, c);
        data_[getIndex(r, c)] = val;
    }

    T Get(size_t r, size_t c) const {
        checkBounds(r, c);
        return data_[getIndex(r, c)];
    }

    T* Flatten() {
        return data_;
    }

    const T* Flatten() const {
        return data_;
    }

    // ** MODIFIED: Unflatten now performs a DEEP COPY and retains ownership **
    void Unflatten(const T* src, size_t r, size_t c) {
        // 1. Clean up existing memory
        cleanup(); 

        if (src == nullptr || r == 0 || c == 0) {
            return;
        }

        // 2. Allocate new internal memory (Deep Copy)
        rows_ = r;
        cols_ = c;
        size_t size = r * c;
        
        data_ = new T[size]; // Allocate NEW memory
        std::copy(src, src + size, data_); // Copy data from external source (src)
    }
};

#endif // MATRIX_HPP