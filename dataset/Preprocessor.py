import pandas as pd

# to run, "python dataset/Preprocessor.py"

# configuration
train_perc = 0.8
random_seed = 0  # seed
target_col = "median_house_value"

# load dataset
df = pd.read_csv("dataset/housing.csv")

# drop samples with NaN values
df = df.dropna()

# drop categorical feature (better to one-hot, but dropped for simplicity)
df = df.drop("ocean_proximity", axis=1)

# shuffle dataset
df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# calculate split index
split_index = int(train_perc * len(df_shuffled))

# separate train and test
train_df = df_shuffled.iloc[:split_index]
test_df = df_shuffled.iloc[split_index:]

# separate features and target
X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]

X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

# calculate min-max on training data
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)

# scale
X_train_scaled = (X_train - X_min) / (X_max - X_min)
X_test_scaled = (X_test - X_min) / (X_max - X_min)

# calculate min-max for targets
y_min = y_train.min()
y_max = y_train.max()
y_range = y_max - y_min

# Scale targets
y_train_scaled = (y_train - y_min) / y_range
y_test_scaled = (y_test - y_min) / y_range

# save CSV
X_train_scaled.to_csv("dataset/X_train_scaled.csv", index=False)
y_train_scaled.to_csv("dataset/y_train_scaled.csv", index=False)

X_test_scaled.to_csv("dataset/X_test_scaled.csv", index=False)
y_test_scaled.to_csv("dataset/y_test_scaled.csv", index=False)