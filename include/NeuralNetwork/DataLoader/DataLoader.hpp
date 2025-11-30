#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include "../Matrix.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <chrono>

/**
 * @brief Helper function to read data from a CSV file into a 2D vector.
 *
 * This function handles file opening, parsing lines, and converting string
 * tokens into the specified type T. It assumes data is comma-separated.
 *
 * @tparam T The numeric type (e.g., float, double) for data storage.
 * @param filepath The path to the CSV file.
 * @param has_header Flag to indicate if the first row is a header (it will be skipped).
 * @return std::vector<std::vector<T>> A vector of vectors containing the parsed data.
 * @throws std::runtime_error if the file cannot be opened.
 */
template <typename T>
std::vector<std::vector<T>> readCsvFile(const std::string& filepath, bool has_header) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::vector<std::vector<T>> data;
    std::string line;

    // Skip header row if present
    if (has_header) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<T> row;

        // Loop over tokens (comma-separated values)
        while (std::getline(ss, token, ',')) {
            try {
                // Use stringstream conversion for generic type safety
                T value;
                std::stringstream convert_ss(token);
                if (convert_ss >> value) {
                    row.push_back(value);
                } else {
                    // Handle conversion error (e.g., non-numeric data)
                    throw std::runtime_error("Data conversion error for token: " + token);
                }
            } catch (const std::exception& e) {
                // Rethrow with more context
                throw std::runtime_error("Error processing value '" + token + "' in file " + filepath + ": " + e.what());
            }
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    return data;
}

/**
 * @brief DataLoader class for managing datasets for machine learning training.
 *
 * It provides batching, shuffling, and status tracking for iterative processing.
 *
 * @tparam T The data type of the matrices (e.g., double or float).
 */
template <typename T>
class DataLoader {
public:
    // Pair of Matrices: features (X) and labels (y)
    using Batch = std::pair<Matrix<T>, Matrix<T>>;

    /**
     * @brief Constructor for the DataLoader.
     * @param batch_size The number of samples per batch.
     */
    DataLoader(size_t batch_size) : batch_size_(batch_size), current_index_(0), total_samples_(0) {}

    /**
     * @brief Loads features (X) and labels (y) from two CSV files.
     * @param features_path Path to the features CSV file.
     * @param labels_path Path to the labels CSV file.
     * @param features_has_header Flag if the features file has a header (default: true).
     * @param labels_has_header Flag if the labels file has a header (default: true).
     * @throws std::runtime_error if file loading fails or data dimensions don't match.
     */
    void loadCSV(const std::string& features_path, const std::string& labels_path, 
                 bool features_has_header = true, bool labels_has_header = true) {
        
        std::cout << "Loading features from: " << features_path << std::endl;
        auto features_data_2d = readCsvFile<T>(features_path, features_has_header);
        
        std::cout << "Loading labels from: " << labels_path << std::endl;
        auto labels_data_2d = readCsvFile<T>(labels_path, labels_has_header);

        if (features_data_2d.empty() || labels_data_2d.empty()) {
            throw std::runtime_error("One or both CSV files are empty or contained only a header.");
        }

        if (features_data_2d.size() != labels_data_2d.size()) {
            throw std::runtime_error("Feature and label files must have the same number of samples (rows).");
        }

        total_samples_ = features_data_2d.size();
        size_t feature_cols = features_data_2d[0].size();
        size_t label_cols = labels_data_2d[0].size();

        // 1. Flatten and load Features Matrix
        std::vector<T> features_flat;
        features_flat.reserve(total_samples_ * feature_cols);
        for (const auto& row : features_data_2d) {
            features_flat.insert(features_flat.end(), row.begin(), row.end());
        }
        features_.Unflatten(features_flat.data(), total_samples_, feature_cols);
        std::cout << "Features loaded: " << total_samples_ << " rows, " << feature_cols << " columns." << std::endl;

        // 2. Flatten and load Labels Matrix
        std::vector<T> labels_flat;
        labels_flat.reserve(total_samples_ * label_cols);
        for (const auto& row : labels_data_2d) {
            labels_flat.insert(labels_flat.end(), row.begin(), row.end());
        }
        labels_.Unflatten(labels_flat.data(), total_samples_, label_cols);
        std::cout << "Labels loaded: " << total_samples_ << " rows, " << label_cols << " columns." << std::endl;

        // 3. Initialize shuffle indices
        shuffled_indices_.resize(total_samples_);
        std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);
        current_index_ = 0; // Reset index after loading new data
    }

    /**
     * @brief Returns the next batch of features and labels.
     *
     * @return Batch A pair containing the features Matrix and the labels Matrix for the batch.
     * @throws std::out_of_range if getBatch is called after the dataset is finished.
     */
    Batch getBatch() {
        if (isFinished()) {
            throw std::out_of_range("Dataset fully processed. Call shuffle() to start a new epoch.");
        }

        // Determine the actual size of the current batch (handles the last, smaller batch)
        size_t batch_end_index = std::min(current_index_ + batch_size_, total_samples_);
        size_t actual_batch_size = batch_end_index - current_index_;
        size_t feature_cols = features_.cols();
        size_t label_cols = labels_.cols();
        
        // Data buffers for the new batch matrices
        std::vector<T> batch_features_data(actual_batch_size * feature_cols);
        std::vector<T> batch_labels_data(actual_batch_size * label_cols);

        // Copy data into the batch buffers using shuffled indices
        for (size_t i = 0; i < actual_batch_size; ++i) {
            size_t original_row_index = shuffled_indices_[current_index_ + i];
            
            // Copy features row
            for (size_t c = 0; c < feature_cols; ++c) {
                batch_features_data[i * feature_cols + c] = features_.Get(original_row_index, c);
            }

            // Copy labels row
            for (size_t c = 0; c < label_cols; ++c) {
                batch_labels_data[i * label_cols + c] = labels_.Get(original_row_index, c);
            }
        }

        // Create new Matrix objects and load the batch data
        Matrix<T> batch_features;
        batch_features.Unflatten(batch_features_data.data(), actual_batch_size, feature_cols);

        Matrix<T> batch_labels;
        batch_labels.Unflatten(batch_labels_data.data(), actual_batch_size, label_cols);

        // Advance the current index for the next batch
        current_index_ += actual_batch_size;

        return {batch_features, batch_labels};
    }

    /**
     * @brief Shuffles the dataset indices for a new epoch and resets the batch index.
     */
    void shuffle() {
        if (total_samples_ == 0) {
            std::cout << "Warning: Cannot shuffle an empty dataset." << std::endl;
            return;
        }
        
        // Use a high-quality random number generator engine
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine engine(seed);
        
        // Shuffle the indices vector
        std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), engine);
        
        // Reset the index to start a new epoch
        current_index_ = 0;
    }

    /**
     * @brief Checks if all samples in the dataset have been processed.
     * @return bool True if the current index is past the total number of samples.
     */
    bool isFinished() const {
        return current_index_ >= total_samples_;
    }

    /**
     * @brief Returns the total number of samples in the dataset.
     */
    size_t totalSamples() const {
        return total_samples_;
    }
    
    /**
     * @brief Returns the batch size set for this DataLoader.
     */
    size_t getBatchSize() const {
        return batch_size_;
    }


private:
    Matrix<T> features_;
    Matrix<T> labels_;
    size_t batch_size_;
    size_t current_index_;
    size_t total_samples_;
    std::vector<size_t> shuffled_indices_; // Used to map shuffled order back to original data
};

#endif // DATALOADER_HPP