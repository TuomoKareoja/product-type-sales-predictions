import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO load data
prediction_path_rf = os.path.join("data", "predictions", "prediction_rf.csv")
prediction_path_glmnet = os.path.join("data", "predictions", "prediction_glmnet.csv")
orig_pred_rf = pd.read_csv(prediction_path_rf, index=False)
orig_pred_glmnet = pd.read_csv(prediction_path_glmnet, index=False)

# Keep only interesting categories smartphone, netbooks, pc, laptop

# TODO plot save path
plot_path = os.path.join("reports", "figures")

# TODO barplot of the products by volume
# order by predicted volume and colorcode

# TODO scatterplot facet of reviews and predicted volume
# X reviews and y volume
# Color and shapecode observations

# Random Forest model uses only some of the review columns
review_features_rf = [
    "rew_4star",
    "rew_3star",
    "rew_2star",
    "rew_1star",
    "positive_service_review",
    "product_type_GameConsole",
]

