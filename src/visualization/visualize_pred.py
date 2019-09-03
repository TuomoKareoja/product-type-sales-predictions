import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# load data
prediction_path_rf = os.path.join("data", "predictions", "prediction_rf.csv")
raw_train_path = os.path.join("data", "raw", "existingproductattributes2017.csv")
orig_pred_rf = pd.read_csv(prediction_path_rf)
raw_train = pd.read_csv(raw_train_path)

# Keep only interesting categories smartphone, netbooks, pc, laptop
interesting_cats = ["Netbook", "PC", "Smartphone", "Laptop"]
orig_pred_rf = orig_pred_rf[orig_pred_rf.ProductType.isin(interesting_cats)]
raw_train = raw_train[raw_train.ProductType.isin(interesting_cats)]

plot_path = os.path.join("reports", "figures")

# barplot of the products by volume
fig, ax = plt.subplots()

sns.barplot(
    y="ProductNum",
    x="Volume",
    data=orig_pred_rf,
    orient="h",
    hue="ProductType",
    dodge=False,
    order=orig_pred_rf.sort_values(by=["Volume"], ascending=False)["ProductNum"],
    ax=ax,
)
ax.set(
    title="Predicted Volume by Product Number and Type\n(Random Forest)",
    xlabel="Predicted Volume",
    ylabel="Product Number",
)
plt.legend(title="Product Type")
fig.savefig(os.path.join(plot_path, "predicted_volume_rf.png"))

reviews_used_in_rf = [
    "x4StarReviews",
    "x3StarReviews",
    "x2StarReviews",
    "x1StarReviews",
    "PositiveServiceReview",
]

reviews_used_in_rf_printable = [
    "4 Star Reviews",
    "3 Star Reviews",
    "2 Star Reviews",
    "1 Star Reviews",
    "Positive Service Reviews",
]

for review, name in zip(reviews_used_in_rf, reviews_used_in_rf_printable):
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=review,
        y="Volume",
        data=raw_train,
        ax=ax,
        label="Old Products",
        color="b",
        s=100,
        alpha=0.7,
    )
    sns.scatterplot(
        x=review,
        y="Volume",
        data=orig_pred_rf,
        ax=ax,
        label="New Products (Predicted)",
        color="r",
        s=100,
        alpha=0.7,
    )
    ax.set(
        title="Association Between " + name + " and Volume\n(Random Forest)",
        xlabel="Number of " + name,
        ylabel="Volume",
    )
    plot_title = "volume_" + review + "_relation.png"
    fig.savefig(os.path.join(plot_path, plot_title))
