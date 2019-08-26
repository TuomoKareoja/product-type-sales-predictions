"""
.. module:: build_features.py
    :synopsis:

"""

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

# Splitting the data to train and predicting set
data_train = data[data.train]
data_pred = data[~data.train]

# we don't need the train column anymore
data_train.drop(columns=["train"], inplace=True)
data_pred.drop(columns=["train"], inplace=True)

# we don't need the target volume in the predict dataset
data_pred.drop(columns=["volume"], inplace=True)

# Saving to disk
data_processed_path_train = os.path.join("data", "processed", "processed_train.csv")
data_processed_path_pred = os.path.join("data", "processed", "processed_pred.csv")
data_train.to_csv(data_processed_path_train, index=False)
data_pred.to_csv(data_processed_path_pred, index=False)
