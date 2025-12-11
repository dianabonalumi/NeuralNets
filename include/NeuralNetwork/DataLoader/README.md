# Data Processing Module: DataLoader

This module provides the necessary infrastructure for efficient data loading, validation, and management during the training of Machine Learning models. The core component, `DataLoader`, abstracts away file handling, data shuffling, and the crucial step of batch creation, ensuring memory-efficient and iterative processing of datasets for model training.

## Module Structure

The file contained in this module is:

* **`DataLoader.hpp`**: Defines the `readCsvFile` helper function and the main `DataLoader` class.

---

## Component Details: `DataLoader.hpp`

### 1. Data Parsing Helper: `readCsvFile`

This templated function handles low-level I/O operations and type conversion.

* **Functionality:** Reads data from a specified CSV file, parsing comma-separated values (tokens) into the numeric type $T$. It includes error checking for file opening and handles potential data conversion errors.
* **Header Handling:** The function supports skipping the first row if a header is present.

### 2. The `DataLoader<T>` Class

This class manages the lifecycle of the dataset (features $X$ and labels $y$) during training epochs, ensuring data is delivered efficiently and randomly in batches.

#### A. Data Loading and Storage

The `loadCSV` method reads two separate files and prepares them for training:

* **Input Validation:** It enforces a critical constraint: the features and labels files must have the same number of samples (rows) to ensure dataset integrity.
* **Matrix Storage:** The parsed 2D data is flattened into 1D vectors and stored internally in two **`Matrix<T>`** objects (`features_` and `labels_`) using the `Unflatten` method, which is optimized for subsequent matrix operations.
* **Index Initialization:** A vector of indices (`shuffled_indices_`) is initialized sequentially ($0, 1, 2, ...$). This vector is key, as it allows for efficient **shuffling** by only manipulating indices, rather than moving the large data matrices themselves.

#### B. Core Training Mechanisms (Batching and Shuffling)

The `DataLoader` implements the essential functions for controlling the SGD training loop:

| Method | Role | Details |
| :--- | :--- | :--- |
| `getBatch()` | **Batch Creation** | Returns the next batch of data (features $X$ and labels $y$) as a `Batch` pair. It uses the shuffled indices to extract rows, calculating the actual size of the batch (which handles the last, potentially smaller batch correctly). |
| `shuffle()` | **Epoch Reset** | **Randomly shuffles** the order of the **`shuffled_indices_`** vector and resets the `current_index_` to zero. This is performed at the start of each epoch, introducing the necessary **stochasticity** for SGD. It uses a system-clock seeded random engine for high-quality shuffling. |
| `isFinished()` | **Epoch Status** | Returns `true` when the `current_index_` has processed all `total_samples_`, signaling the completion of an epoch. |

---

## External Dependencies

* **`Matrix.hpp`**: The fundamental data structure used to store and manage the feature and label data internally. The `DataLoader` relies on `Unflatten` for loading and `Get(r, c)` for precise batch extraction.
* **Standard Library**: The module utilizes `<fstream>` and `<sstream>` for robust file reading, and `<random>`, `<algorithm>` (`std::shuffle`), and `<chrono>` for implementing the high-quality shuffling mechanism.