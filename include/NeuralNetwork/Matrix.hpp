#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <stdexcept>
#include <algorithm>

template <typename T>
class Matrix {
private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    T* data_ = nullptr;
    bool owns_data_ = true;

    void cleanup() {
        if (owns_data_ && data_ != nullptr) {
            delete[] data_;
        }
        data_ = nullptr;
        rows_ = 0;
        cols_ = 0;
        // Setting owns_data_ back to true here is slightly misleading for a 
        // completely cleaned state, but maintains the invariant that if data_
        // is nullptr, it behaves as if it's owned (or empty).
        owns_data_ = true; 
    }

    size_t getIndex(size_t r, size_t c) const {
        // Row-major indexing
        return r * cols_ + c;
    }

    void checkBounds(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Matrix indices out of bounds.");
        }
    }

public:
    // ** FIX: Added default constructor **
    // This allows Matrix to be default-constructed (e.g., as a member of another class).
    Matrix() = default; // 
    
    size_t getRows() const { return rows_; }
    size_t getCols() const { return cols_; }

    // Destructor (Rule of Three/Five/Zero)
    ~Matrix() {
        cleanup();
    }

    // Constructor: Allocates and owns memory
    Matrix(size_t r, size_t c) : rows_(r), cols_(c) {
        if (r == 0 || c == 0) {
            data_ = nullptr;
            rows_ = 0;
            cols_ = 0;
        } else {
            size_t size = r * c;
            // Value initialization {} ensures elements are zeroed (e.g., to 0 for int/double)
            data_ = new T[size] {}; 
            owns_data_ = true;
        }
    }

    // Sets the value at (r, c)
    void Set(size_t r, size_t c, const T& val) {
        checkBounds(r, c);
        data_[getIndex(r, c)] = val;
    }

    // Gets the value at (r, c)
    T Get(size_t r, size_t c) const {
        checkBounds(r, c);
        return data_[getIndex(r, c)];
    }

    // Returns a mutable pointer to the start of the flattened data
    T* Flatten() {
        return data_;
    }

    // Returns a constant pointer to the start of the flattened data
    const T* Flatten() const {
        return data_;
    }

    // Unflatten: Points to external memory (non-owning view)
    void Unflatten(T* src, size_t r, size_t c) {
        // 1. Clean up existing memory if owned by this matrix
        cleanup();

        // 2. Point to the external memory without copying
        rows_ = r;
        cols_ = c;
        data_ = src;
        owns_data_ = false; // Important: we do not own this memory
    }
};

#endif // MATRIX_HPP