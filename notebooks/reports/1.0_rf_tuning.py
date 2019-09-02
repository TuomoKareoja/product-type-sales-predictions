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
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFECV

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.visualization.visualize import plot_cv_predictions
from sklearn.model_selection import RandomizedSearchCV


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

# dropping the best seller rank, because it contains missing values and does not
# seem to help
X.drop(columns=["bestseller_rank"], inplace=True)

#%%

# Defining the method for crossvalidation. We crossvalidate each individual row
crossvalidation = KFold(n_splits=70, shuffle=True, random_state=seed)

# Defining list of scoring methods
scoring = ["neg_mean_squared_error", "neg_mean_absolute_error"]

#%%

# lets first try the model without any optimization or feature selection

pipelines = []
pipelines.append(("RF", make_pipeline(RandomForestRegressor(random_state=seed))))

plot_cv_predictions(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="unoptimized",
)

#%%

# lets optimize the hyperparameters with random search and see if we get better results

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
}

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=seed),
    param_distributions=random_grid,
    n_iter=50,
    # leaving 2 rows out
    cv=35,
    verbose=2,
    random_state=seed,
    n_jobs=-1,
)

rf_random.fit(X, y)


rf_best_parameters = rf_random.best_params_

#%%

# Fitting the model with best found parameters

pipelines = []
pipelines.append(("RF", make_pipeline(RandomForestRegressor(**rf_best_parameters))))

plot_cv_predictions(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="optimized",
)

# parameter optimization clearly reduced the amount of overfitting

#%%

# lets add feature selection to the mix. Random forest is normally quite good against
# overfitting but separate feature selection step might still make the model better

# lets do this by using the variables that are most import in the optimized model
# with all the predictors

rf_model = RandomForestRegressor(**rf_best_parameters)
rf_model.fit(X, y)
importances = rf_model.feature_importances_

# x_values = list(range(len(importances)))# Make a bar chart
plt.bar(X.columns, importances, orientation="vertical")  # Tick labels for x axis
plt.ylabel("Importance")
plt.xlabel("Variable")
plt.title("Variable Importances")
plt.xticks(rotation="vertical")

#%%

# Keeping only the star reviews, positive service reviews and product type game console

most_important_feats = [
    "rew_4star",
    "rew_3star",
    "rew_2star",
    "rew_1star",
    "positive_service_review",
    "product_type_GameConsole",
]

X_mif = X[most_important_feats]

#%%

# optimizing the hyperparameters with this feature selection

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
}

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=seed),
    param_distributions=random_grid,
    n_iter=50,
    # leaving 2 rows out
    cv=35,
    verbose=2,
    random_state=seed,
    n_jobs=-1,
)

rf_random.fit(X_mif, y)

rf_best_parameters = rf_random.best_params_

#%%

# Fitting the model with best found parameters and only selected features

pipelines = []
pipelines.append(("RF_mif", make_pipeline(RandomForestRegressor(**rf_best_parameters))))

plot_cv_predictions(
    pipelines=pipelines,
    X=X_mif,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="optimized",
)


#%%

# Lets also try a different method for feature selection by using RFE and linear selector

estimator = LinearRegression()
scoring = "neg_mean_absolute_error"
rfecv_selector = RFECV(estimator, step=1, cv=crossvalidation, scoring=scoring)
rfecv_selector = rfecv_selector.fit(X, y)

X.columns[rfecv_selector.support_]
rfecv_selector.grid_scores_
rfecv_selector.support_

X_rfe = X.iloc[:, rfecv_selector.support_]

# optimizing the hyperparameters for rfe

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
}

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=seed),
    param_distributions=random_grid,
    n_iter=50,
    # leaving 2 rows out
    cv=35,
    verbose=2,
    random_state=seed,
    n_jobs=-1,
)

rf_random.fit(X_rfe, y)

rf_best_parameters = rf_random.best_params_

#%%

# Fitting the model with best found parameters and only selected features

pipelines = []
pipelines.append(("RF_rfe", make_pipeline(RandomForestRegressor(**rf_best_parameters))))

plot_cv_predictions(
    pipelines=pipelines,
    X=X_rfe,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="optimized",
)

# the model based tuning seems to work much better.


#%%

# Lets build the final model with the features selected from the model
# This we give the random tuning more iterations as this is the final model

# Keeping only the star reviews, positive service reviews and product type game console

most_important_feats = [
    "rew_4star",
    "rew_3star",
    "rew_2star",
    "rew_1star",
    "positive_service_review",
    "product_type_GameConsole",
]

X_mif = X[most_important_feats]

#%%

# optimizing the hyperparameters with this feature selection

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
}

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=seed),
    param_distributions=random_grid,
    n_iter=150,
    # leaving 2 rows out
    cv=35,
    verbose=2,
    random_state=seed,
    n_jobs=-1,
)

rf_random.fit(X_mif, y)

rf_best_parameters = rf_random.best_params_

#%%

# Fitting the model with best found parameters and only selected features

pipelines = []
pipelines.append(("RF_mif", make_pipeline(RandomForestRegressor(**rf_best_parameters))))

plot_cv_predictions(
    pipelines=pipelines,
    X=X_mif,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="best",
)


#%%

