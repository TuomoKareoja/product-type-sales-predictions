from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import os

# Load data
data_train_path = os.path.join("data", "raw", "existingproductattributes2017.csv")
data_pred_path = os.path.join("data", "raw", "newproductattributes2017.csv")
data_train = pd.read_csv(data_train_path)
data_pred = pd.read_csv(data_pred_path)

# Combining the data for easier handling and renaming columns
data_train["train"] = True
data_pred["train"] = False

# setting predicted volume to missing instead of 0
data_pred["Volume"] = np.nan

# Dropping extended warranties from the training dataset
# These observations seem to broken as their feature values
# are almost identical
data_train.query('ProductType != "ExtendedWarranty"', inplace=True)

# Renaming columns
data = pd.concat([data_train, data_pred])
data.columns = [
    "product_type",
    "product_num",
    "price",
    "rew_5star",
    "rew_4star",
    "rew_3star",
    "rew_2star",
    "rew_1star",
    "positive_service_review",
    "negative_service_review",
    "recommend_product",
    "bestseller_rank",
    "weight",
    "depth",
    "width",
    "height",
    "profit_margin",
    "volume",
    "train",
]

# The product number column is not necessary so we drop it
data.drop(columns=["product_num"], inplace=True)

# The bestseller rank contains missing values and is hard to fill so we drop it
data.drop(columns=["bestseller_rank"], inplace=True)

# Drop 5 star reviews as they have perfect correlation with sales volume. Impossible!
data.drop(columns=["rew_5star"], inplace=True)

# mark depth, width and height 0 as missing
data[["weight", "depth", "width"]].replace(0, np.nan, inplace=True)

# saving to /clean
data_clean_path = os.path.join("data", "clean", "clean_train_pred.csv")
data.to_csv(data_clean_path, index=False)
