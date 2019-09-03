import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import copy

# Importing libraries
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline

import sys

from ..visualization.visualize_cv import plot_cv_predictions

seed = 42

X_train_path = os.path.join("data", "processed", "X_train.csv")
y_train_path = os.path.join("data", "processed", "y_train.csv")
X_predict_path = os.path.join("data", "processed", "X_pred.csv")
orig_predict_path = os.path.join("data", "raw", "newproductattributes2017.csv")
X = pd.read_csv(X_train_path)
y = pd.read_csv(y_train_path, index_col=False)
X_pred = pd.read_csv(X_predict_path)
orig_pred = pd.read_csv(orig_predict_path)
y = y.iloc[:, 0]

# Defining the method for crossvalidation. We crossvalidate each individual row
crossvalidation = KFold(n_splits=70, shuffle=True, random_state=seed)

# Defining list of scoring methods
scoring = ["neg_mean_absolute_error"]


######################
##  Random Forests  ##
######################

# Keeping only the most important features. Check rf_tuning.py for details.
most_important_feats_rf = [
    "rew_4star",
    "rew_3star",
    "rew_2star",
    "rew_1star",
    "positive_service_review",
    "product_type_GameConsole",
]

X_rf = X[most_important_feats_rf]

# Check rf_tuning for how these were found
rf_best_parameters = {
    "n_estimators": 2000,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "max_depth": None,
}

# Pipeline to put models in
rf_pipeline_list = []
rf_pipeline_list.append(
    ("RF", make_pipeline(RandomForestRegressor(**rf_best_parameters)))
)


##############
##  GLMNET  ##
##############

most_important_feats_glmnet = [
    "rew_4star",
    "rew_3star",
    "rew_2star",
    "negative_service_review",
    "recommend_product",
    "width",
    "profit_margin",
    "product_type_Accessories",
    "product_type_GameConsole",
    "product_type_Laptop",
    "product_type_Netbook",
    "product_type_PC",
    "product_type_Printer",
    "product_type_PrinterSupplies",
    "product_type_Smartphone",
    "product_type_Software",
    "product_type_Tablet",
]

X_glmnet = X[most_important_feats_glmnet]

glmnet = ElasticNetCV(cv=crossvalidation, random_state=seed)
glmnet = glmnet.fit(X_glmnet, y)
glmnet_best_params = glmnet.get_params()

glmnet_model = ElasticNet(random_state=seed)
glmnet_best_params_matching = {
    key: glmnet_best_params[key]
    for key in glmnet_model.get_params().keys()
    if key in glmnet_best_params
}

# manual tuning so that things work
glmnet_best_params_matching["precompute"] = False
glmnet_best_params_matching

# Pipeline to put models in
glmnet_pipeline_list = []
glmnet_pipeline_list.append(
    ("GLMNET", make_pipeline(ElasticNet(**glmnet_best_params_matching)))
)

###########################################
##  Plotting Crossvalidated Predictions  ##
###########################################

plot_cv_predictions(
    pipelines=rf_pipeline_list,
    X=X_rf,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="final",
    show=False,
)

plot_cv_predictions(
    pipelines=glmnet_pipeline_list,
    X=X_glmnet,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="final",
    show=False,
)

is_pc = X.product_type_PC == 1
is_laptop = X.product_type_Laptop == 1
is_netbook = X.product_type_Netbook == 1
is_smartphone = X.product_type_Smartphone == 1

limited_cat = ((is_pc) | (is_laptop) | (is_netbook) | (is_smartphone)).to_list()

plot_cv_predictions(
    pipelines=rf_pipeline_list,
    X=X_rf,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="final_lim_pred",
    limited_pred_mask=limited_cat,
    show=False,
)

plot_cv_predictions(
    pipelines=glmnet_pipeline_list,
    X=X_glmnet,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="final_lim_pred",
    limited_pred_mask=limited_cat,
    show=False,
)


#####################
##  Final Training ##
#####################

rf = RandomForestRegressor(**rf_best_parameters)
glmnet = ElasticNet(**glmnet_best_params_matching)

rf.fit(X_rf, y)
glmnet.fit(X_glmnet, y)


############################
##  Predicting and saving ##
############################

X_pred_rf = X_pred[most_important_feats_rf]
X_pred_glmnet = X_pred[most_important_feats_glmnet]
orig_pred_rf = copy.deepcopy(orig_pred)
orig_pred_glmnet = copy.deepcopy(orig_pred)

orig_pred_rf["Volume"] = rf.predict(X_pred_rf)
orig_pred_glmnet["Volume"] = glmnet.predict(X_pred_glmnet)

prediction_path_rf = os.path.join("data", "predictions", "prediction_rf.csv")
prediction_path_glmnet = os.path.join("data", "predictions", "prediction_glmnet.csv")
orig_pred_rf.to_csv(prediction_path_rf, index=False)
orig_pred_glmnet.to_csv(prediction_path_glmnet, index=False)
