#%%

# Importing libraries
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import find_dotenv, load_dotenv
from IPython.core.interactiveshell import InteractiveShell

from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.visualization.visualize import plot_cv_scores
from src.visualization.visualize import plot_cv_predictions

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

# creatingpipelines for model testing

pipelines = []

# simple imputation
pipelines.append(
    (
        "SVM_simple",
        make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy="mean"),
            preprocessing.StandardScaler(),
            LinearSVR(C=4, random_state=seed),
        ),
    )
)
pipelines.append(
    (
        "RF_simple",
        make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy="mean"),
            preprocessing.StandardScaler(),
            RandomForestRegressor(n_estimators=100, random_state=seed),
        ),
    )
)
pipelines.append(
    (
        "KNN_simple",
        make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy="mean"),
            preprocessing.StandardScaler(),
            KNeighborsRegressor(),
        ),
    )
)
pipelines.append(
    (
        "LM_simple",
        make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy="mean"),
            preprocessing.StandardScaler(),
            LinearRegression(),
        ),
    )
)

# ExtraTreesRegressor
pipelines.append(
    (
        "SVM_extratrees",
        make_pipeline(
            IterativeImputer(ExtraTreesRegressor(n_estimators=10, random_state=0)),
            preprocessing.StandardScaler(),
            LinearSVR(C=4, random_state=seed),
        ),
    )
)
pipelines.append(
    (
        "RF_extratrees",
        make_pipeline(
            IterativeImputer(ExtraTreesRegressor(n_estimators=10, random_state=0)),
            preprocessing.StandardScaler(),
            RandomForestRegressor(n_estimators=100, random_state=seed),
        ),
    )
)
pipelines.append(
    (
        "KNN_extratrees",
        make_pipeline(
            IterativeImputer(ExtraTreesRegressor(n_estimators=10, random_state=0)),
            SimpleImputer(),
            preprocessing.StandardScaler(),
            KNeighborsRegressor(),
        ),
    )
)
pipelines.append(
    (
        "LM_extratrees",
        make_pipeline(
            IterativeImputer(ExtraTreesRegressor(n_estimators=10, random_state=0)),
            preprocessing.StandardScaler(),
            LinearRegression(),
        ),
    )
)

# BayesianRidge
pipelines.append(
    (
        "SVM_bayesridge",
        make_pipeline(
            IterativeImputer(BayesianRidge()),
            preprocessing.StandardScaler(),
            LinearSVR(C=4, random_state=seed),
        ),
    )
)
pipelines.append(
    (
        "RF_bayesridge",
        make_pipeline(
            IterativeImputer(BayesianRidge()),
            preprocessing.StandardScaler(),
            RandomForestRegressor(n_estimators=100, random_state=seed),
        ),
    )
)
pipelines.append(
    (
        "KNN_bayesridge",
        make_pipeline(
            IterativeImputer(BayesianRidge()),
            preprocessing.StandardScaler(),
            KNeighborsRegressor(),
        ),
    )
)
pipelines.append(
    (
        "LM_bayesridge",
        make_pipeline(
            IterativeImputer(BayesianRidge()),
            preprocessing.StandardScaler(),
            LinearRegression(),
        ),
    )
)

#%%

plot_cv_scores(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    scoring=scoring,
    file_suffix="unoptimized",
)

plot_cv_predictions(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="unoptimized",
)

# most of the error in the models comes from a small amount of outliers.
# In some cases the models are quite unaccurate

# With the unoptimized models random forest looks to be
# the best as it scores better in neg_mean_squared_error and new_mean_absolute_error
# and its predictions are clearly more accurate

# it seems that the imputation is not helping and with extratrees it is actually
# hurting.

# If we compare our models to ones we created in task 2 the performance
# is much worse. This is probably because in that exercise we still used the
# 5 star reviews that have perfect correlation with the target variable

#%%

# Because imputation did not seem to help much we try to just drop that
# bestseller_rank (the only variable with missing values) and see if we
# get better results or even if the results stay the same

X_no_bestseller = X.drop(columns="bestseller_rank")

pipelines = []

pipelines.append(
    (
        "SVM_no_bestseller",
        make_pipeline(
            preprocessing.StandardScaler(), LinearSVR(C=4, random_state=seed)
        ),
    )
)
pipelines.append(
    (
        "RF_no_bestseller",
        make_pipeline(
            preprocessing.StandardScaler(),
            RandomForestRegressor(n_estimators=100, random_state=seed),
        ),
    )
)
pipelines.append(
    (
        "KNN_no_bestseller",
        make_pipeline(preprocessing.StandardScaler(), KNeighborsRegressor()),
    )
)
pipelines.append(
    (
        "LM_no_bestseller",
        make_pipeline(preprocessing.StandardScaler(), LinearRegression()),
    )
)

#%%

# plot_cv_scores(
#     pipelines=pipelines,
#     X=X_no_bestseller,
#     y=y,
#     crossvalidation=crossvalidation,
#     scoring=scoring,
#     file_suffix="unoptimized",
# )

plot_cv_predictions(
    pipelines=pipelines,
    X=X_no_bestseller,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="unoptimized",
)

# For all models dropping the variable results in lower out of sample
# prediction error

# Checking if dropping the product category also results in better results

X_no_cat_no_bestseller = X.drop(
    columns=[
        "bestseller_rank",
        "product_type_Accessories",
        "product_type_Display",
        "product_type_ExtendedWarranty",
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
        "SVM_no_cat_no_bestseller",
        make_pipeline(
            preprocessing.StandardScaler(), LinearSVR(C=4, random_state=seed)
        ),
    )
)
pipelines.append(
    (
        "RF_no_cat_no_bestseller",
        make_pipeline(
            preprocessing.StandardScaler(),
            RandomForestRegressor(n_estimators=100, random_state=seed),
        ),
    )
)
pipelines.append(
    (
        "KNN_no_cat_no_bestseller",
        make_pipeline(preprocessing.StandardScaler(), KNeighborsRegressor()),
    )
)
pipelines.append(
    (
        "LM_no_cat_no_bestseller",
        make_pipeline(preprocessing.StandardScaler(), LinearRegression()),
    )
)

#%%

plot_cv_scores(
    pipelines=pipelines,
    X=X_no_cat_no_bestseller,
    y=y,
    crossvalidation=crossvalidation,
    scoring=scoring,
    file_suffix="unoptimized",
)

plot_cv_predictions(
    pipelines=pipelines,
    X=X_no_cat_no_bestseller,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="unoptimized",
)

# Dropping the product category makes all the models perform better
#%%

# Checking if dropping 3 and 1 star reviews helps. There features are highly correlated with
# 4 and 2 star reviews respectively, but have a smaller correlation with volume

X_no_cat_no_bestseller_less_stars = X.drop(
    columns=[
        "bestseller_rank",
        "product_type_Accessories",
        "product_type_Display",
        "product_type_ExtendedWarranty",
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
        "SVM_no_cat_no_bestseller_less_stars",
        make_pipeline(
            preprocessing.StandardScaler(), LinearSVR(C=4, random_state=seed)
        ),
    )
)
pipelines.append(
    (
        "RF_no_cat_no_bestseller_less_stars",
        make_pipeline(
            preprocessing.StandardScaler(),
            RandomForestRegressor(n_estimators=100, random_state=seed),
        ),
    )
)
pipelines.append(
    (
        "KNN_no_cat_no_bestseller_less_stars",
        make_pipeline(preprocessing.StandardScaler(), KNeighborsRegressor()),
    )
)
pipelines.append(
    (
        "LM_no_cat_no_bestseller_less_stars",
        make_pipeline(preprocessing.StandardScaler(), LinearRegression()),
    )
)

#%%

plot_cv_scores(
    pipelines=pipelines,
    X=X_no_cat_no_bestseller_less_stars,
    y=y,
    crossvalidation=crossvalidation,
    scoring=scoring,
    file_suffix="unoptimized",
)

plot_cv_predictions(
    pipelines=pipelines,
    X=X_no_cat_no_bestseller_less_stars,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="unoptimized",
)

# Dropping the start makes all models except knn perform better.

#%%

# On the basis of this explorations we decide to do the following things

# 1. Dropping futher testing of KNN and SVM models as these perform baddly. Continuing
# to more precise testing with RF and LM
# 2. We need to use some feature selection algoritm to as dropping features seems to create
# a lot of value
