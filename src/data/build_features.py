from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import os

# Loading clean data
data_clean_path = os.path.join("data", "clean", "clean_train_pred.csv")
data = pd.read_csv(data_clean_path)

# one-hot-encoding the product type
data = pd.get_dummies(data)

# Dropping the extended warranty column as there are zero
# warranties in the training set
data.drop(columns=["product_type_ExtendedWarranty"], inplace=True)

# Splitting the data to train and predicting set
data_train = data[data.train]
data_pred = data[~data.train]

# we don't need the train column anymore
data_train.drop(columns=["train"], inplace=True)
data_pred.drop(columns=["train"], inplace=True)

# we don't need the target volume in the predict dataset
data_pred.drop(columns=["volume"], inplace=True)

# Separating target and features to different datasets
X_train = data_train.drop(columns=["volume"])
y_train = data_train[["volume"]]
X_pred = data_pred

# Saving to disk
data_processed_path_X_train = os.path.join("data", "processed", "X_train.csv")
data_processed_path_y_train = os.path.join("data", "processed", "y_train.csv")
data_processed_path_X_pred = os.path.join("data", "processed", "X_pred.csv")
X_train.to_csv(data_processed_path_X_train, index=False)
y_train.to_csv(data_processed_path_y_train, index=False)
X_pred.to_csv(data_processed_path_X_pred, index=False)
