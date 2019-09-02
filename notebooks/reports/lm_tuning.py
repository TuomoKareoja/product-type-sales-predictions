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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFECV

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

# feature selection affected linear model greatly so lets automate this process
# by using instead of the normal LinearRegression library lets use ElasticNet
# that combines L1 and L2 regularization. This works as a kind of automatic feature
# selection

# lets start by optimizing the parameters in crossvalidation. There is a separate
# function for this package that does this more efficiently

glmnet = ElasticNetCV(cv=70, random_state=seed)
glmnet.fit(X, y)
glmnet_best_params = glmnet.get_params()

#%%

# Defining the method for crossvalidation. We crossvalidate each individual row
crossvalidation = KFold(n_splits=70, shuffle=True, random_state=seed)

# Defining list of scoring methods
scoring = ["neg_mean_squared_error", "neg_mean_absolute_error"]

#%%

glmnet_model = ElasticNet()
glmnet_best_params_matching = {
    key: glmnet_best_params[key]
    for key in glmnet_model.get_params().keys()
    if key in glmnet_best_params
}

# manual tuning so that things work
glmnet_best_params_matching["precompute"] = False

pipelines = []

pipelines.append(("GLMNET", make_pipeline(ElasticNet(**glmnet_best_params_matching))))

#%%

plot_cv_predictions(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="optimized",
)

# The model is performing quite poorly and the results are still not as good as with the
# regular lm model with manual (uninformed) variable dropping.

#%%

glmnet_best_params_matching

# We can see that the l1-l2 ratio is 0.5, so a perfect mixture.
# To get better and more strict variable selection, lets try to force
# a more strict feature selection by using full lasso regression (only l1 regularization)

#%%

glmnet_model = ElasticNet()
glmnet_best_params_matching = {
    key: glmnet_best_params[key]
    for key in glmnet_model.get_params().keys()
    if key in glmnet_best_params
}

# manual tuning so that things work
glmnet_best_params_matching["precompute"] = False

# forcing only l1 regularization
glmnet_best_params_matching["l1_ratio"] = 1

pipelines = []

pipelines.append(("Lasso", make_pipeline(ElasticNet(**glmnet_best_params_matching))))

#%%

plot_cv_predictions(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="optimized",
)

# The model performs now even worse in the outsample even though the insample error is smaller

#%%

# Plotting the volume histogram to see if a transformation could help
sns.distplot(y, bins=20)

#%%
# Taking the log of the target makes it much more normally distributed
# this should lessen the effects of extreme values

sns.distplot(np.log(y).clip(lower=0), bins=20)

#%%

# Lets transform y to log and continue with that
# We need to apply lower 0 because there are some zero values and these would
# be -inf if we don't do this

y_log = np.log(y).clip(lower=0)

#%%

# lets try to do the same as before but this time with a transformed y-variable

# lets start by optimizing the parameters in crossvalidation.

glmnet = ElasticNetCV(cv=70, random_state=seed)
glmnet.fit(X, y_log)
glmnet_best_params = glmnet.get_params()

#%%

# Defining list of scoring methods
scoring = ["neg_mean_squared_error", "neg_mean_absolute_error"]

glmnet_model = ElasticNet()
glmnet_best_params_matching = {
    key: glmnet_best_params[key]
    for key in glmnet_model.get_params().keys()
    if key in glmnet_best_params
}

# manual tuning so that things work
glmnet_best_params_matching["precompute"] = False

pipelines = []

pipelines.append(("GLMNET", make_pipeline(ElasticNet(**glmnet_best_params_matching))))

plot_cv_predictions(
    pipelines=pipelines,
    X=X,
    y=y_log,
    crossvalidation=crossvalidation,
    file_suffix="optimized_log",
    transformation="exp",
    round_digits=2,
)

glmnet_best_params_matching

#%%

glmnet_model = ElasticNet()
glmnet_best_params_matching = {
    key: glmnet_best_params[key]
    for key in glmnet_model.get_params().keys()
    if key in glmnet_best_params
}

# manual tuning so that things work
glmnet_best_params_matching["precompute"] = False

# forcing only l1 regularization
glmnet_best_params_matching["l1_ratio"] = 1

pipelines = []

pipelines.append(("Lasso", make_pipeline(ElasticNet(**glmnet_best_params_matching))))

plot_cv_predictions(
    pipelines=pipelines,
    X=X,
    y=y_log,
    crossvalidation=crossvalidation,
    file_suffix="optimized_log",
    transformation="exp",
    round_digits=2,
)

glmnet_best_params_matching


#%%

# Both models have one huge outlier in their predictions. Lets try to find out what
# observation this is

predicted_outsample = cross_val_predict(pipelines[0][1], X, y_log, cv=crossvalidation)

test = list(np.exp(y_log) - np.exp(predicted_outsample))
X.iloc[test.index(min(test))]
y.iloc[test.index(min(test))]

# It is an accessory with lots of good reviews and excellent sales.
# Seems like a legit observations. Good sales and even better reviews.

#%%

# Trying out a recursive feature elimination with crossvalidation
# and running these variables trough the glmnet optimization and just normal linear model

estimator = LinearRegression()
scoring = "neg_mean_absolute_error"
rfecv_selector = RFECV(estimator, step=1, cv=crossvalidation, scoring=scoring)
rfecv_selector = rfecv_selector.fit(X, y)

X.columns
X.columns[rfecv_selector.support_]
rfecv_selector.grid_scores_
rfecv_selector.support_

X_rfe = X.iloc[:, rfecv_selector.support_]

glmnet = ElasticNetCV(cv=crossvalidation, random_state=seed)
glmnet = glmnet.fit(X_rfe, y)
glmnet_best_params = glmnet.get_params()

# Defining list of scoring methods
scoring = ["neg_mean_squared_error", "neg_mean_absolute_error"]

glmnet_model = ElasticNet()
glmnet_best_params_matching = {
    key: glmnet_best_params[key]
    for key in glmnet_model.get_params().keys()
    if key in glmnet_best_params
}

# manual tuning so that things work
glmnet_best_params_matching["precompute"] = False

pipelines = []

pipelines.append(("GLMNET", make_pipeline(ElasticNet(**glmnet_best_params_matching))))

pipelines.append(("LM", make_pipeline(LinearRegression())))

plot_cv_predictions(
    pipelines=pipelines,
    X=X_rfe,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="optimized_rfe",
)

glmnet_best_params_matching

# Now the model performs better than any of our manual efforts for feature selection
# with linear models. We can also see that the glmnet performs bit better than the simple
# linear model even after we have done aggressive feature selection.

#%%

# lets build and tune a final linear model. We use the rfe feature selection with
# crossvalidated glmnet with a the l1 ratio decided by this process

estimator = LinearRegression()
scoring = "neg_mean_absolute_error"
rfecv_selector = RFECV(estimator, step=1, cv=crossvalidation, scoring=scoring)
rfecv_selector = rfecv_selector.fit(X, y)

X_rfe = X.iloc[:, rfecv_selector.support_]

glmnet = ElasticNetCV(cv=crossvalidation, random_state=seed)
glmnet = glmnet.fit(X_rfe, y)
glmnet_best_params = glmnet.get_params()

# Defining list of scoring methods
scoring = ["neg_mean_squared_error", "neg_mean_absolute_error"]

glmnet_model = ElasticNet(random_state=seed)
glmnet_best_params_matching = {
    key: glmnet_best_params[key]
    for key in glmnet_model.get_params().keys()
    if key in glmnet_best_params
}

# manual tuning so that things work
glmnet_best_params_matching["precompute"] = False
glmnet_best_params_matching

pipelines = []

pipelines.append(("GLMNET", make_pipeline(ElasticNet(**glmnet_best_params_matching))))

plot_cv_predictions(
    pipelines=pipelines,
    X=X_rfe,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="best_rfe",
)


#%%

# Trying how the model handels when we limit the predictions to only interesting product types

is_pc = X.product_type_PC == 1
is_laptop = X.product_type_Laptop == 1
is_netbook = X.product_type_Netbook == 1
is_smartphone = X.product_type_Smartphone == 1

y_lim_pred = y[(is_pc) | (is_laptop) | (is_netbook) | (is_smartphone)]

pipelines = []
pipelines.append(("GLMNET", make_pipeline(ElasticNet(**glmnet_best_params_matching))))

path_figures = os.path.join("reports", "figures")

for name, pipeline in pipelines:
    predicted_outsample = cross_val_predict(pipeline, X_rfe, y, cv=crossvalidation)
    predicted_insample = pipeline.fit(X_rfe, y).predict(X_rfe)

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

# Looks quite bad as the some of the predictions are actually negative

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
X_lim_pred = X_rfe[(is_pc) | (is_laptop) | (is_netbook) | (is_smartphone)]

# We need to define our crossvalidation again to match the more limited number
# of observations
crossvalidation = KFold(n_splits=13, shuffle=True, random_state=seed)

pipelines = []
pipelines.append(("GLMNET", make_pipeline(ElasticNet(**glmnet_best_params_matching))))

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


#%%

