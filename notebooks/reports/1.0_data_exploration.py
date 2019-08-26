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

# Setting styles
# %matplotlib inline
sns.set(style="ticks", color_codes=True)
# InteractiveShell.ast_node_interactivity = "all"

#%%

# loading in data
data_train_path = os.path.join("data", "raw", "existingproductattributes2017.csv")
data_train = pd.read_csv(data_train_path)
data_train.head()

#%%

# loading in data
data_pred_path = os.path.join("data", "raw", "newproductattributes2017.csv")
data_pred = pd.read_csv(data_pred_path)
data_pred.head()

#%%

# combining the data for easier handling and renaming columns

data_train["train"] = True
data_pred["train"] = False

# setting predicted volume to missing instead of 0
data_pred["Volume"] = np.nan

data = pd.concat([data_train, data_pred])
data.columns = [
    "product_type",
    "product_num",
    "price",
    "rew_5star",
    "rew_4star",
    "rew_3star",
    "rew_2star",
    "rew_1star",
    "positive_service_review",
    "negative_service_review",
    "recommend_product",
    "bestseller_rank",
    "weight",
    "depth",
    "width",
    "height",
    "profit_margin",
    "volume",
    "train",
]

data.head()

#%%

# Categorical variables

# all the other variables
plot_data_train = pd.DataFrame(
    data[data.train].product_type.value_counts()
).sort_index()
plot_data_test = pd.DataFrame(
    data[data.train != True].product_type.value_counts()
).sort_index()
fig, ax = plt.subplots()
ax.bar(plot_data_train.index, plot_data_train.product_type, label="train")
ax.bar(
    plot_data_test.index,
    plot_data_test.product_type,
    label="test",
    bottom=plot_data_train.product_type,
)
ax.set_xticklabels(plot_data_train.index, rotation=90)
ax.set_ylabel("Number of Products")
ax.legend()
ax.set_title("Distribution of Product Types")
plt.show()

#%%


def draw_swarmplot(x, y, data):
    fig, ax = plt.subplots()
    ax = sns.swarmplot(x=x, y=y, data=data)
    ax = sns.boxplot(
        x=x,
        y=y,
        data=data,
        showcaps=False,
        boxprops={"facecolor": "None", "linewidth": 1},
        showfliers=False,
        whiskerprops={"linewidth": 0},
    )
    ax.set_title(y)
    plt.xticks(rotation=90)
    plt.show()
    plt.close()


#%%
draw_swarmplot("product_type", "price", data)
#%%
draw_swarmplot("product_type", "rew_5star", data)
#%%
draw_swarmplot("product_type", "rew_4star", data)
#%%
draw_swarmplot("product_type", "rew_3star", data)
#%%
draw_swarmplot("product_type", "rew_2star", data)
#%%
draw_swarmplot("product_type", "rew_1star", data)
#%%
draw_swarmplot("product_type", "positive_service_review", data)
#%%
draw_swarmplot("product_type", "negative_service_review", data)
#%%
draw_swarmplot("product_type", "recommend_product", data)
#%%
draw_swarmplot("product_type", "bestseller_rank", data)
#%%
draw_swarmplot("product_type", "weight", data)
#%%
draw_swarmplot("product_type", "depth", data)
#%%
draw_swarmplot("product_type", "width", data)
#%%
draw_swarmplot("product_type", "height", data)
#%%
draw_swarmplot("product_type", "profit_margin", data)
#%%

# This takes a long time and the results are very interesting so saving this to disk
pairplot = sns.pairplot(data.select_dtypes(exclude=["object"]), hue="train")
pairplot.savefig(os.path.join("reports", "figures", "numerical_pairplot.png"))

#%%

# drawing correlation matrix
corr = data.select_dtypes(include=["int64", "float64"]).corr()
corr.style.background_gradient(cmap="coolwarm", axis=None)

# 5 star reviews have a perfect correlation to volume. This cannot not be right
# 4 and 3 star reviews have very high correlation. 3 star reviews could be dropped
# if multicollinearity starts to be problem, because it has lower correlation
# with volume. Same with 2 and 1 star reviews

# product number does not correlate with anything. This was just to check
# if the would have been some coding errors in the data. We can drop this
# feature as useless

#%%

# are there products that have more reviews than sales?
data_plot = data[data.train == True]
data_plot["reviews"] = (
    data_plot.rew_5star
    + data_plot.rew_4star
    + data_plot.rew_3star
    + data_plot.rew_2star
    + data_plot.rew_1star
)

fig, ax = plt.subplots()
ax.scatter(data_plot.volume, data_plot.reviews)
ax.set(xlim=(0, 12000), ylim=(0, 12000))
ax.set_xlabel("Volume")
ax.set_ylabel("Reviews")
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

for i, product_id in enumerate(data_plot.product_num.astype(object)):
    # print(product_id, data_plot.volume[i], data_plot.reviews[i])
    ax.annotate(product_id, xy=(data_plot["volume"][i], data_plot["reviews"][i]))

#%%

# product number 123 has more reviews than sales. This seems to mostly because
# of a huge number of 1 star reviews. Probably review bombing?
# We should probably just drop this product it is really difficult the
# the decide which review or reviews are wrong. Or we could assign all
# review values as missing
data.loc[data.product_num == 123].head()

# UPDATE this is probably okay, because as we have reviews from products that we have not sold
# yet (the observations to predict), the reviews must have been gathered from an external
# review site. Because this is the case reviews and sales don't have to match

#%%

# From old knowledge we know that the product category 'extended warranty'
# seems to have repeated observations, that have just a different price.
# We are going to need to deal with these somehow, maybe group by all other
# variables and use the median of the prices
data.loc[data.product_type == "ExtendedWarranty"]

#%%

# checking out missing
data.isnull().mean().multiply(100).round(0)
# we have lots of missing values in bestseller_rank. Maybe we should create a model
# to input these?

#%%

# checking if something should be missing but is not
# counting the number of values missing or zero
data.fillna(0).astype(bool).mean().multiply(100).round(0)

#%%
# depth and width have zero values but there should now be any
data[(data.depth == 0) | (data.width == 0) | (data.height == 0)]

# All of these products have weight so depth, width and height are
# clearly just missing and not really 0
# One of these is actually in the test set
