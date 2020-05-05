import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from model.tests_data import tests_areas, tests_actual
from preprocessing.process_data import import_polygons_data, calculate_area


def get_feature_names(coeffs, features):

    if len(coeffs) % len(features) == 0:
        scaling = int(len(coeffs) / len(features))
        index = []

        for i in range(1, scaling + 1):
            feature_names = [col.replace('_', ' ') + f"_{i}" for col in features]
            index.extend(feature_names)
    else:
        return None

    return index


def get_feature_importance_df(model, features):

    feature_names = get_feature_names(model.feature_importances_, features)

    if not feature_names:
        return

    importance_df = pd.DataFrame(index=feature_names)
    importance_df['importance'] = model.feature_importances_
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df


def get_coefficients_df(reg, features):

    feature_names = get_feature_names(reg.coef_, features)

    if not feature_names:
        return

    coefs_df = pd.DataFrame(index=feature_names)
    coefs_df['coefs'] = reg.coef_
    coefs_df['abs_coefs'] = np.abs(reg.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)

    return coefs_df


def plot_results(results, factor=100.0):
    """
    Plots the results
    :param results: a data frame with the results
    :param factor: factor to multiply the results with
    :param cmap: color map of the heatmap
    :param cbar: whether to display a color bar
    :return: None
    """

    plt.figure(figsize=(5, 10))

    sb.heatmap(
        results*factor,
        annot=True,
        fmt=".2f",
        cmap='Blues',
        cbar_kws={'format': '%.0f%%'},
        linecolor='#f2f2f2',
        linewidth=0.1
    )

    plt.show()


def plot_comparisons(results, factor=100.0, cbar_kws=None):

    if not cbar_kws:
        cbar_kws = {'format': '%.0f%%'}

    mask = np.ones(results.shape)
    mask[:, -1] = False

    plt.figure(figsize=(8, 10))

    ax = sb.heatmap(
        results * factor,
        vmin=results.values[:, -1].ravel().min() * factor,
        vmax=results.values[:, -1].ravel().max() * factor,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap='Blues',
        cbar_kws=cbar_kws,
        linecolor='#f2f2f2',
        linewidths=0.01
    )

    for (j, i), label in np.ndenumerate(results.values):
        if i != results.shape[1] - 1:

            label_number = int(label)

            ax.text(i+0.5, j+0.5, "{:.2e}".format(label_number) if label_number > 1E4 else label_number,
                    fontdict=dict(ha='center',  va='center',
                                  color='black', fontsize=10))

    plt.show()


def model_evaluation(model, X_test, y_test, features, population_tests_dir, grid_search=False, plot=True):

    if grid_search:
        logging.info(f"best estimator: {model.best_estimator_}")
        logging.info(f"best params: {model.best_params_}")
        logging.info(f"best score: {model.best_score_}")
        # logging.info(f"all_results: {model.cv_results_}")

        return

    pruned_features = [feature for feature in features if feature != 'area']

    if isinstance(model['reg'], (LinearRegression, Ridge, Lasso)):
        results_df = get_coefficients_df(model['reg'], pruned_features)
        factor = 1
        cbar_kws = {'format': '%.0f'}

        if plot:
            plot_comparisons(results_df[:20], factor, cbar_kws)
    else:
        results_df = get_feature_importance_df(model['reg'], pruned_features)
        factor = 100

        if plot:
            plot_results(results_df[:20], factor)

    total_score = model.score(X_test, y_test)

    logging.info(f'Total score: {total_score}')

    population_tests, file_names = import_polygons_data(population_tests_dir)

    population_results = {
        "prediction": {},
        "actual": {},
        "difference": {}
    }

    for i in range(population_tests.shape[0]):

        test_case = file_names[i]

        test_row = population_tests.iloc[i:i+1, :].copy()

        if test_case in tests_areas:
            test_row["area"] = tests_areas[test_case]

        else:
            test_row["area"] = test_row["geometry"].apply(lambda poly: calculate_area(poly))

        pred = model.predict(test_row) * test_row["area"]

        population_results["prediction"][test_case] = pred.values[0]
        population_results["actual"][test_case] = tests_actual[test_case] \
            if test_case in tests_actual else None

        try:
            diff = (population_results["prediction"][test_case] - population_results["actual"][test_case]) \
                   / population_results["actual"][test_case]
            population_results["difference"][test_case] = round(diff, 2)

        except TypeError:
            population_results["difference"][test_case] = 'N/A'

    results_df = pd.DataFrame(population_results)

    if plot:
        plot_comparisons(results_df)
