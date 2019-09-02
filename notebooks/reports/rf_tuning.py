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

#%%

# Defining the method for crossvalidation. We crossvalidate each individual row
crossvalidation = KFold(n_splits=70, shuffle=True, random_state=seed)

# Defining list of scoring methods
scoring = ["neg_mean_squared_error", "neg_mean_absolute_error"]

#%%

# Defining hyperparameter search area

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

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=seed),
    param_distributions=random_grid,
    n_iter=100,
    cv=10,
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

plt.bar(X.columns, importances, orientation="vertical")
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

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=seed),
    param_distributions=random_grid,
    n_iter=100,
    # leaving 2 rows out
    cv=10,
    verbose=2,
    random_state=seed,
    n_jobs=-1,
)

rf_random.fit(X_mif, y)

rf_best_parameters = rf_random.best_params_

#%%

# Fitting the model with best found parameters and only selected features

pipelines = []
pipelines.append(("RF", make_pipeline(RandomForestRegressor(**rf_best_parameters))))

plot_cv_predictions(
    pipelines=pipelines,
    X=X_mif,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="optimized_mif",
)

#%%

# Lets build the final model with the features selected from the model
# This we give the random tuning more iterations as this is the final model

# optimizing the hyperparameters with this feature selection

rf_random = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=seed),
    param_distributions=random_grid,
    n_iter=150,
    # leaving 2 rows out
    cv=10,
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

# The insample error went down but the outsample error rose. It seems that there is
# actually not enough data to do reliable hyperparameter optimization

#%%

# Trying how the model handels when we limit the predictions to only somewhat related
# product types

# Dropping rows that are not interesting for our predictions
is_pc = X.product_type_PC == 1
is_laptop = X.product_type_Laptop == 1
is_netbook = X.product_type_Netbook == 1
is_smartphone = X.product_type_Smartphone == 1

y_lim_pred = y[(is_pc) | (is_laptop) | (is_netbook) | (is_smartphone)]

pipelines = []
pipelines.append(("RF", make_pipeline(RandomForestRegressor(**rf_best_parameters))))

path_figures = os.path.join("reports", "figures")

for name, pipeline in pipelines:
    predicted_outsample = cross_val_predict(pipeline, X_mif, y, cv=crossvalidation)
    predicted_insample = pipeline.fit(X_mif, y).predict(X_mif)

    predicted_outsample = predicted_outsample[
        (is_pc) | (is_laptop) | (is_netbook) | (is_smartphone)
    ]

    predicted_insample = predicted_insample[
        (is_pc) | (is_laptop) | (is_netbook) | (is_smartphone)
    ]
    fig, ax = plt.subplots()
    ax.scatter(y_lim_pred, predicted_outsample, c="r", marker="+", label="outsample")
    ax.scatter(y_lim_pred, predicted_insample, c="b", marker="x", label="insample")
    ax.plot(
        [y_lim_pred.min(), y_lim_pred.max()],
        [y_lim_pred.min(), y_lim_pred.max()],
        "k--",
        lw=3,
    )
    model_mean_absolute_error_outsample = round(
        metrics.mean_absolute_error(y_lim_pred, predicted_outsample), 0
    )
    model_mean_absolute_error_insample = round(
        metrics.mean_absolute_error(y_lim_pred, predicted_insample), 0
    )
    plt.text(
        0.05,
        0.9,
        "Mean absolute error (outsample): " + str(model_mean_absolute_error_outsample),
        transform=ax.transAxes,
    )
    plt.text(
        0.05,
        0.8,
        "Mean absolute error (insample): " + str(model_mean_absolute_error_insample),
        transform=ax.transAxes,
    )
    ax.set_xlabel("Actual Volume")
    ax.set_ylabel("Predicted Volume")
    plt.legend(loc=4)
    fig.suptitle(name + ": " + "Predicted vs Actual")
    plot_title = "predictions_" + "best_lim_pred" + "_" + name.lower() + ".png"
    plt.savefig(os.path.join(path_figures, plot_title))
    plt.show()
    plt.clf()

#%%

# Testing how things change if we use only the relevant product types also for training
# We don't bother to try to do hyperparameter optimization for this dataset
# as it is just too small to do this reliably


# Dropping rows that are not interesting for our predictions
is_pc = X.product_type_PC == 1
is_laptop = X.product_type_Laptop == 1
is_netbook = X.product_type_Netbook == 1
is_smartphone = X.product_type_Smartphone == 1

y_lim_pred = y[(is_pc) | (is_laptop) | (is_netbook) | (is_smartphone)]
X_lim_pred = X_mif[(is_pc) | (is_laptop) | (is_netbook) | (is_smartphone)]

# We need to define our crossvalidation again to match the more limited number
# of observations
crossvalidation = KFold(n_splits=13, shuffle=True, random_state=seed)

pipelines = []
pipelines.append(("RF", make_pipeline(RandomForestRegressor(**rf_best_parameters))))

path_figures = os.path.join("reports", "figures")

for name, pipeline in pipelines:
    predicted_outsample = cross_val_predict(
        pipeline, X_lim_pred, y_lim_pred, cv=crossvalidation
    )
    predicted_insample = pipeline.fit(X_lim_pred, y_lim_pred).predict(X_lim_pred)
    fig, ax = plt.subplots()
    ax.scatter(y_lim_pred, predicted_outsample, c="r", marker="+", label="outsample")
    ax.scatter(y_lim_pred, predicted_insample, c="b", marker="x", label="insample")
    ax.plot(
        [y_lim_pred.min(), y_lim_pred.max()],
        [y_lim_pred.min(), y_lim_pred.max()],
        "k--",
        lw=3,
    )
    model_mean_absolute_error_outsample = round(
        metrics.mean_absolute_error(y_lim_pred, predicted_outsample), 0
    )
    model_mean_absolute_error_insample = round(
        metrics.mean_absolute_error(y_lim_pred, predicted_insample), 0
    )
    plt.text(
        0.05,
        0.9,
        "Mean absolute error (outsample): " + str(model_mean_absolute_error_outsample),
        transform=ax.transAxes,
    )
    plt.text(
        0.05,
        0.8,
        "Mean absolute error (insample): " + str(model_mean_absolute_error_insample),
        transform=ax.transAxes,
    )
    ax.set_xlabel("Actual Volume")
    ax.set_ylabel("Predicted Volume")
    plt.legend(loc=4)
    fig.suptitle(name + ": " + "Predicted vs Actual")
    plot_title = "predictions_" + "best_lim_pred_train" + "_" + name.lower() + ".png"
    plt.savefig(os.path.join(path_figures, plot_title))
    plt.show()
    plt.clf()

# When we use the other data the model becomes better

#%%
