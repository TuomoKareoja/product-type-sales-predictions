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
from sklearn.model_selection import KFold

from sklearn.pipeline import make_pipeline
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

# dropping the best seller rank, because it contains missing values and does not
# seem to help
X.drop(columns=["bestseller_rank"], inplace=True)

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

plot_cv_scores(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    scoring=scoring,
    file_suffix="optimized",
)

plot_cv_predictions(
    pipelines=pipelines,
    X=X,
    y=y,
    crossvalidation=crossvalidation,
    file_suffix="optimized",
)


#%%
