"""
.. module:: visualize.py
    :synopsis:

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


def add_values(bp, ax):
    """Adds the numbers to the various points of the boxplots"""
    for element in ["medians"]:
        for line in bp[element]:
            # Get the position of the element. y is the label you want
            (x_l, y), (x_r, _) = line.get_xydata()
            # Make sure datapoints exist
            if not np.isnan(y):
                x_line_center = x_l + (x_r - x_l) / 2
                y_line_center = y  # Since it's a line and it's horizontal
                # overlay the value:  on the line, from center to right
                ax.text(
                    x_line_center,
                    y_line_center,  # Position
                    "%.3f" % y,  # Value (3f = 3 decimal float)
                    verticalalignment="center",  # Centered vertically with line
                    fontsize=6,
                )


def plot_cv_scores(pipelines, X, y, crossvalidation, scoring, file_suffix=""):
    """Plots crossvalidation scores with given scoring methods and pipeline
    
    Arguments:
        pipelines {list} -- list of sklearn pipelines
        X {dataframe} -- Features in a dataframe format
        y {series} -- Target feature
        crossvalidation {Kfold object} -- KFold crossvalidation pattern
        scoring {list} -- list of scoring methods to use
 
    Keyword Arguments:
        file_suffix {str} -- suffix to add to plot names to distinguish them (default: {""})
    """

    path_figures = os.path.join("reports", "figures")

    for score_method in scoring:
        results = []
        names = []
        for name, pipeline in pipelines:
            cv_results = cross_val_score(
                pipeline, X, y, cv=crossvalidation, scoring=score_method
            )
            results.append(cv_results)
            names.append(name)
            msg = "(%s) %s: %f (%f)" % (
                score_method,
                name,
                cv_results.mean(),
                cv_results.std(),
            )
            print(msg)
        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle("Estimator Comparison: " + score_method)
        ax = fig.add_subplot(111)
        bp_dict = plt.boxplot(results)
        add_values(bp_dict, ax)
        ax.set_xticklabels(names)
        ax.set_ylabel(score_method)
        plot_title = (
            "model_comparison_" + file_suffix + "_" + score_method.lower() + ".png"
        )
        plt.savefig(os.path.join(path_figures, plot_title), dpi=200)
        plt.show()
        plt.clf()


def plot_cv_predictions(pipelines, X, y, crossvalidation, file_suffix=""):
    """Plots crossvalidation predictions with given pipelines
    
    Arguments:
        pipelines {list} -- list of sklearn pipelines
        X {dataframe} -- Features in a dataframe format
        y {series} -- Target feature
        crossvalidation {Kfold object} -- KFold crossvalidation pattern
   
    Keyword Arguments:
        file_suffix {str} -- suffix to add to plot names to distinguish them (default: {""})
    """

    path_figures = os.path.join("reports", "figures")

    for name, pipeline in pipelines:
        predicted = cross_val_predict(pipeline, X, y, cv=crossvalidation)
        fig, ax = plt.subplots()
        ax.scatter(y, predicted, edgecolors=(0, 0, 0))
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=3)
        fig.suptitle(name + ": " + "Predicted vs Actual")
        ax.set_xlabel("Actual Volume")
        ax.set_ylabel("Predicted Volume")
        model_mean_absolute_error = round(metrics.mean_absolute_error(y, predicted))
        plt.text(
            0.2,
            0.8,
            "Mean absolute error: " + str(model_mean_absolute_error),
            transform=ax.transAxes,
        )
        plot_title = "predictions_" + file_suffix + "_" + name.lower() + ".png"
        plt.savefig(os.path.join(path_figures, plot_title))
        plt.show()
        plt.clf()
