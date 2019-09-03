#%%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVR

from src.visualization.visualize_cv import plot_cv_predictions, plot_cv_scores

# Setting styles
# %matplotlib inline
sns.set(style="ticks", color_codes=True)
InteractiveShell.ast_node_interactivity = "all"

seed = 42

#%% # Loading the data
X_train_path = os.path.join("data", "processed", "X_train.csv")
y_train_path = os.path.join("data", "processed", "y_train.csv")
X = pd.read_csv(X_train_path)
y = pd.read_csv(y_train_path, index_col=False)
y = y.iloc[:, 0]

#%%

# Defining the method for crossvalidation. We cross validate individual values
crossvalidation = KFold(n_splits=70, shuffle=True, random_state=seed)

# Defining list of scoring methods
scoring = ["neg_mean_squared_error", "neg_mean_absolute_error"]


#%%

pipelines = []

pipelines.append(
    (
        "SVM",
        make_pipeline(
            preprocessing.StandardScaler(), LinearSVR(C=4, random_state=seed)
        ),
    )
)
pipelines.append(
    ("RF", make_pipeline(RandomForestRegressor(n_estimators=100, random_state=seed)))
)
pipelines.append(
    ("KNN", make_pipeline(preprocessing.StandardScaler(), KNeighborsRegressor()))
)
pipelines.append(("LM", make_pipeline(LinearRegression())))

#%%

plot_cv_scores(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    scoring=scoring,
    file_suffix="unoptimized_simple",
)

plot_cv_predictions(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="unoptimized_simple",
)

#%%

# Checking if dropping the product category results in better predictions

X_no_cat = X.drop(
    columns=[
        "product_type_Accessories",
        "product_type_Display",
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
)

pipelines = []

pipelines.append(
    (
        "SVM",
        make_pipeline(
            preprocessing.StandardScaler(), LinearSVR(C=4, random_state=seed)
        ),
    )
)
pipelines.append(
    (
        "RF",
        make_pipeline(
            preprocessing.StandardScaler(),
            RandomForestRegressor(n_estimators=100, random_state=seed),
        ),
    )
)
pipelines.append(
    ("KNN", make_pipeline(preprocessing.StandardScaler(), KNeighborsRegressor()))
)
pipelines.append(
    ("LM", make_pipeline(preprocessing.StandardScaler(), LinearRegression()))
)

#%%

plot_cv_scores(
    pipelines=pipelines,
    X=X_no_cat,
    y=y,
    crossvalidation=crossvalidation,
    scoring=scoring,
    file_suffix="unoptimized_no_cat",
)

plot_cv_predictions(
    pipelines=pipelines,
    X=X_no_cat,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="unoptimized_no_cat",
)

# Dropping the product category makes all the models perform better
#%%

# Checking if dropping 3 and 1 star reviews helps. There features are highly correlated with
# 4 and 2 star reviews respectively, but have a smaller correlation with volume

X_no_cat_less_stars = X.drop(
    columns=[
        "product_type_Accessories",
        "product_type_Display",
        "product_type_GameConsole",
        "product_type_Laptop",
        "product_type_Netbook",
        "product_type_PC",
        "product_type_Printer",
        "product_type_PrinterSupplies",
        "product_type_Smartphone",
        "product_type_Software",
        "product_type_Tablet",
        "rew_3star",
        "rew_1star",
    ]
)

pipelines = []

pipelines.append(
    (
        "SVM",
        make_pipeline(
            preprocessing.StandardScaler(), LinearSVR(C=4, random_state=seed)
        ),
    )
)
pipelines.append(
    (
        "RF",
        make_pipeline(
            preprocessing.StandardScaler(),
            RandomForestRegressor(n_estimators=100, random_state=seed),
        ),
    )
)
pipelines.append(
    ("KNN", make_pipeline(preprocessing.StandardScaler(), KNeighborsRegressor()))
)
pipelines.append(
    ("LM", make_pipeline(preprocessing.StandardScaler(), LinearRegression()))
)

#%%

plot_cv_scores(
    pipelines=pipelines,
    X=X_no_cat_less_stars,
    y=y,
    crossvalidation=crossvalidation,
    scoring=scoring,
    file_suffix="unoptimized_no_cat_less_stars",
)

plot_cv_predictions(
    pipelines=pipelines,
    X=X_no_cat_less_stars,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="unoptimized_no_cat_less_stars",
)

# Dropping the start makes all models except knn perform better.

#%%

# On the basis of this explorations we decide to do the following things

# 2. Dropping futher testing of KNN and SVM models as these perform badly. Continuing
# to more precise testing with RF and LM. LM is still bad but it is interesting for
# its interpretability
# 3. We need to use some feature selection algorithm, as dropping features seems to create
# a lot of value
