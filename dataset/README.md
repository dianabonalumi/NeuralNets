## Data Preprocessing Folder

This folder contains the Python script used for preparing the machine learning dataset, as well as the processed and scaled data files ready for model training and evaluation.

The core script is `Preprocessor.py`.

---

### `Preprocessor.py`

This script handles the data loading, cleaning, splitting, and scaling for the California Housing machine learning task. The script performs the following steps on the source file (`dataset/housing.csv`):

1.  **Load Data:** Loads the California Housing dataset using the `pandas` library.
2.  **Handle Missing Values:** Removes any samples (rows) where values are missing (`NaN`).
3.  **Feature Exclusion:** Drops the categorical feature `"ocean_proximity"`.
4.  **Shuffle and Split:** The dataset is shuffled using a fixed `random_seed` (0) and then split into training (80%) and testing (20%) sets based on the `train_perc` configuration.
5.  **Separate Features and Target:** Isolates the features ($\mathbf{X}$) from the target variable ($\mathbf{y}$), which is `"median_house_value"`.
6.  **Scaling (Min-Max Normalization):**
    * **Process:** Min-Max scaling is applied to both features and the target variable to normalize their values into the range $[0, 1]$.
    * **Data Leakage Prevention:** The minimum ($\mathbf{X}_{\text{min, train}}$) and maximum ($\mathbf{X}_{\text{max, train}}$) values are calculated **exclusively from the training data** to ensure the test set remains unseen during the scaling process.
    * **Formula:** The transformation used is:
        $$\mathbf{X}_{\text{scaled}} = \frac{\mathbf{X} - \mathbf{X}_{\text{min, train}}}{\mathbf{X}_{\text{max, train}} - \mathbf{X}_{\text{min, train}}}$$
7.  **Save Processed Data:** The four resulting scaled data frames are saved as new CSV files.

### Configuration

The script's behavior is controlled by these configuration variables:

| Variable | Value | Description |
| :--- | :--- | :--- |
| `train_perc` | `0.8` | The proportion of the dataset reserved for the training set (80%). |
| `random_seed` | `0` | Seed used for shuffling, ensuring identical and reproducible splits. |
| `target_col` | `"median_house_value"` | The name of the column that serves as the dependent variable ($\mathbf{y}$). |

### How to Run

To execute the preprocessing script and generate the scaled data files, navigate to the containing directory and run:

```bash
python Preprocessor.py