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
from sklearn import preprocessing

from sklearn.pipeline import make_pipeline

# Setting styles
# %matplotlib inline
sns.set(style="ticks", color_codes=True)
# InteractiveShell.ast_node_interactivity = "all"

#%%

# Loading the data
data_processed_path_train = os.path.join("data", "processed", "processed_train.csv")
data_processed_path_pred = os.path.join("data", "processed", "processed_test.csv")
data_train = pd.read_csv(data_processed_path_train)
data_pred = pd.read_csv(data_processed_path_pred)

#%%

# creating pipelines for model testing

svm_pipeline = make_pipeline(preprocessing.StandardScaler(), LinearSVR())
rf_pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor())
knn_pipeline = make_pipeline(preprocessing.StandardScaler(), KNeighborsRegressor())
lm_pipeline = make_pipeline(LinearRegression())

#%%
