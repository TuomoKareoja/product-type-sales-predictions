"""
.. module:: clean_data.py
    :synopsis:

"""

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

# Drop 5 star reviews as they have perfect correlation with sales volume. Impossible!
data.drop(columns=["rew_5star"], inplace=True)

# mark depth, width and height 0 as missing
data[["weight", "depth", "width"]].replace(0, np.nan, inplace=True)

# Combining the Extended Warranty products in the train data that have almost
# identical values (most importantly reviews and volume). The problematic
# observations can be recognized from the fact that they have volume 1232.
# We combide these rows to one using the median of the values and the
# lowest product number
fixed_warranty_row = pd.DataFrame(
    data[(data.product_type == "ExtendedWarranty") & (data.volume == 1232)].median()
).transpose()

# Adding back the product type category that was dropping because of the median calculation
fixed_warranty_row["product_type"] = "ExtendedWarranty"

# fixing column types
fixed_warranty_row["product_type"] = fixed_warranty_row["product_type"].astype("object")
# not needed because empty
# fixed_warranty_row["product_num"] = fixed_warranty_row["product_num"].astype("int64")
fixed_warranty_row["price"] = fixed_warranty_row["price"].astype("float64")
fixed_warranty_row["rew_4star"] = fixed_warranty_row["rew_4star"].astype("int64")
fixed_warranty_row["rew_3star"] = fixed_warranty_row["rew_3star"].astype("int64")
fixed_warranty_row["rew_2star"] = fixed_warranty_row["rew_2star"].astype("int64")
fixed_warranty_row["rew_1star"] = fixed_warranty_row["rew_1star"].astype("int64")
fixed_warranty_row["positive_service_review"] = fixed_warranty_row[
    "positive_service_review"
].astype("int64")
fixed_warranty_row["negative_service_review"] = fixed_warranty_row[
    "negative_service_review"
].astype("int64")
fixed_warranty_row["recommend_product"] = fixed_warranty_row[
    "recommend_product"
].astype("float64")
fixed_warranty_row["bestseller_rank"] = fixed_warranty_row["bestseller_rank"].astype(
    "float64"
)
fixed_warranty_row["weight"] = fixed_warranty_row["weight"].astype("float64")
fixed_warranty_row["depth"] = fixed_warranty_row["depth"].astype("float64")
fixed_warranty_row["width"] = fixed_warranty_row["width"].astype("float64")
fixed_warranty_row["height"] = fixed_warranty_row["height"].astype("float64")
fixed_warranty_row["profit_margin"] = fixed_warranty_row["profit_margin"].astype(
    "float64"
)
fixed_warranty_row["volume"] = fixed_warranty_row["volume"].astype("float64")
fixed_warranty_row["train"] = fixed_warranty_row["train"].astype("bool")

# dropping the old row
data.query('product_type != "ExtendedWarranty" and volume != 1232', inplace=True)

# Combining with the modified row
data = pd.concat([data, fixed_warranty_row], copy=False, sort=False)

# saving to /clean
data_clean_path = os.path.join("data", "clean", "clean_train_pred.csv")
data.to_csv(data_clean_path, index=False)
