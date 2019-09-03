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


def plot_cv_scores(
    pipelines, X, y, crossvalidation, scoring, file_suffix="", show=True
):
    """Plots crossvalidation scores with given scoring methods and pipeline
    
    Arguments:
        pipelines {list} -- list of sklearn pipelines
        X {dataframe} -- Features in a dataframe format
        y {series} -- Target feature
        crossvalidation {Kfold object} -- KFold crossvalidation pattern
        scoring {list} -- list of scoring methods to use
 
    Keyword Arguments:
        file_suffix {str} -- suffix to add to plot names to distinguish them (default: {""})
        show {bool} -- print the plot with plt.show() (default: {"True"})
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
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(path_figures, plot_title), dpi=200)
        if show:
            plt.show()
        plt.clf()


def plot_cv_predictions(
    pipelines,
    X,
    y,
    crossvalidation,
    file_suffix="",
    transformation=None,
    round_digits=0,
    limited_pred_mask=None,
    show=True,
):
    """Plots crossvalidation predictions with given pipelines
    
    Arguments:
        pipelines {list} -- list of sklearn pipelines
        X {dataframe} -- Features in a dataframe format
        y {series} -- Target feature
        crossvalidation {Kfold object} -- KFold crossvalidation pattern
   
    Keyword Arguments:
        file_suffix {str} -- suffix to add to plot names to distinguish them (default: {""})
        transformation {str} -- What transformation to apply to predictions after creating them (default: {None})
        round_digits {int} -- How many digits to show in printed error measures (default: {0})
        limited_pred_mask {bool_list} -- Boolean list to filter the shown predictions (default: {False})
        show {bool} -- print the plot with plt.show() (default: {"True"})
    """

    path_figures = os.path.join("reports", "figures")

    for name, pipeline in pipelines:
        predicted_outsample = cross_val_predict(pipeline, X, y, cv=crossvalidation)
        predicted_insample = pipeline.fit(X, y).predict(X)
        if transformation == "exp":
            predicted_insample = np.exp(predicted_insample)
            predicted_outsample = np.exp(predicted_outsample)
            y = np.exp(y)

        if limited_pred_mask is not None:

            y_lim = y[limited_pred_mask]
            predicted_insample = predicted_insample[limited_pred_mask]
            predicted_outsample = predicted_outsample[limited_pred_mask]
        else:
            y_lim = y

        fig, ax = plt.subplots()
        ax.scatter(y_lim, predicted_outsample, c="r", marker="+", label="outsample")
        ax.scatter(y_lim, predicted_insample, c="b", marker="x", label="insample")
        ax.plot([y_lim.min(), y_lim.max()], [y_lim.min(), y_lim.max()], "k--", lw=3)
        model_mean_absolute_error_outsample = round(
            metrics.mean_absolute_error(y_lim, predicted_outsample), round_digits
        )
        model_mean_absolute_error_insample = round(
            metrics.mean_absolute_error(y_lim, predicted_insample), round_digits
        )
        plt.text(
            0.05,
            0.9,
            "Mean absolute error (outsample): "
            + str(model_mean_absolute_error_outsample),
            transform=ax.transAxes,
        )
        plt.text(
            0.05,
            0.8,
            "Mean absolute error (insample): "
            + str(model_mean_absolute_error_insample),
            transform=ax.transAxes,
        )
        ax.set_xlabel("Actual Volume")
        ax.set_ylabel("Predicted Volume")
        plt.legend(loc=4)
        fig.suptitle(name + ": " + "Predicted vs Actual")
        plot_title = "predictions_" + file_suffix + "_" + name.lower() + ".png"
        plt.savefig(os.path.join(path_figures, plot_title))
        if show:
            plt.show()
        plt.clf()
