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

#%%

# Plotting the volume histogram to see if a transformation could help
sns.distplot(y, bins=20)


#%%

# feature selection affected linear model greatly so lets automate this process
# by using instead of the normal LinearRegression library lets use ElasticNet
# that combines L1 and L2 regularization. This works as a kind of automatic feature
# selection

