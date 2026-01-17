# %%
import pandas as pd
import numpy as np

# %%
# file is in arff format
with open("ECG5000/ECG5000_TRAIN.arff", "r") as f:
    train = f.readlines()

with open("ECG5000/ECG5000_TEST.arff", "r") as f:
    test = f.readlines()

with open("ECG5000.csv", "w") as f:
    f.writelines(train[145:])
    f.writelines(test[145:])

# %%
data = pd.read_csv("ECG5000.csv", header=None)
data.describe()

# %%
data.head()

# %%
data_normal = data[data[140] == 1]
data_normal[140].value_counts()

# %%
train_val_test_split = np.arange(data_normal.shape[0])
np.random.seed(0)
np.random.shuffle(train_val_test_split)
train_val_test_split

# %%
train_size = int(len(train_val_test_split) * 0.8)
val_size = int(len(train_val_test_split) * 0.1)

# %%
data_train = data_normal.iloc[train_val_test_split[:train_size]]
data_val = data_normal.iloc[train_val_test_split[train_size:train_size+val_size]]
data_test = data_normal.iloc[train_val_test_split[train_size+val_size:]]

# %%
data_train.drop(140, axis=1).to_csv("train.csv", index=None)
data_train[140].to_csv("train_labels.csv", index=None)
data_val.drop(140, axis=1).to_csv("val.csv", index=None)
data_val[140].to_csv("val_labels.csv", index=None)
data_test.drop(140, axis=1).to_csv("test.csv", index=None)
data_test[140].to_csv("test_labels.csv", index=None)

# %% [markdown]
# Generate classifier dataset

# %%
train_id = train_val_test_split[:train_size]
train_id.shape

# %%
data_class = data_test.copy()
#data_class = data_class.append(data_test)
data_class = data_class.append(data[data[140] != 1])
data_class[140].value_counts()

# %%
data_class.drop(140, axis=1).to_csv("test_classifier.csv", index=None)
data_class[140].to_csv("test_classifier_labels.csv", index=None)


