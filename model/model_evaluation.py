import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import colors, cm

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error

from model.tests_data import tests_actual
from preprocessing.process_data import import_polygons_data, calculate_area


base_color = sb.color_palette()[0]


def remove_borders(fig):
    """
    Removes the borders from a matplotlib figure.

    :param fig: figure whose borders to be removed
    :return: None
    """

    axes = fig.axes[0]
    axes.spines["top"].set_visible(False)
    axes.spines["left"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["bottom"].set_visible(False)


def get_feature_names(coeffs, features):
    """
    Returns the features names according to their polynomial degree

    :param coeffs: coefficients / importances of the model
    :param features: original features array
    :return: The mapped feature names
    """

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
    """
    Returns the feature importance results data frame

    :param model: the fitted model
    :param features: the sequence of the original features
    :return: the results data frame
    """

    feature_names = get_feature_names(model.feature_importances_, features)

    if not feature_names:
        return

    importance_df = pd.DataFrame(index=feature_names)
    importance_df['importance'] = model.feature_importances_
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df


def get_coefficients_df(reg, features):
    """
    Returns the extracted regressor coefficients results on a data frame
    :param reg: the fitted regressor
    :param features: the sequence of the original features
    :return: the results data frame
    """

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

    """
    Plots the city / place population comparisons results as a heatmap

    :param results: a data frame with the results
    :param factor: factor to multiply the results with
    :param cbar_kws: how to format the color bar. Optional
    :return: None
    """

    if not cbar_kws:
        cbar_kws = {'format': '%.0f%%'}

    mask = np.ones(results.shape)
    mask[:, -1] = False

    plt.figure(figsize=(8, 10))

    vmin = results.values[:, -1].ravel().min() * factor
    vmax = results.values[:, -1].ravel().max() * factor

    kwargs = {"cmap": 'Blues'}

    if vmin < 0:
        norm = colors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)

        blues = cm.get_cmap('Blues', 128)

        neg_pos = np.vstack((blues(np.linspace(1, 0, 128)), blues(np.linspace(0, 1, 128))))

        cmap = colors.ListedColormap(neg_pos, name='CustomBlues')

        kwargs.update({"norm": norm, "cmap": cmap})

    ax = sb.heatmap(
        results * factor,
        vmin=vmin,
        vmax=vmax,
        mask=mask,
        annot=True,
        fmt=".1f",
        cbar_kws=cbar_kws,
        linecolor='#f2f2f2',
        linewidths=0.01,
        **kwargs
    )

    for (j, i), label in np.ndenumerate(results.values):
        if i != results.shape[1] - 1:

            label_number = int(label)

            ax.text(i+0.5, j+0.5, "{:.2e}".format(label_number) if label_number > 1E4 else label_number,
                    fontdict=dict(ha='center',  va='center',
                                  color='black', fontsize=10))

    plt.show()


def plot_barplot(column, df, title, x_label, y_label):
    """
    Plots a barplot of the coefficients results

    :param column: column of the values to be plotted
    :param df: data frame
    :param title: title of plot
    :param x_label: x axis label
    :param y_label: y axis label
    :return: None
    """

    fig = plt.figure(figsize=(7, 10))
    sb.barplot(data=df, x=column, y=df.index, color=base_color)

    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    remove_borders(fig)


def plot_feature_results(model, features, plot=True):
    """
    Retrieves and plots the coefficients / feature importance results.

    :param model: fitted model
    :param features: sequence of the original features
    :param plot: whether results should be plotted or not.
    :return: None
    """

    pruned_features = [feature for feature in features if feature != 'area']

    if isinstance(model['reg'], (LinearRegression, Ridge, Lasso)):
        results_df = get_coefficients_df(model['reg'], pruned_features)
        factor = 1
        cbar_kws = {'format': '%.0f'}

        if plot and results_df is not None:

            plot_barplot(
                'coefs',
                results_df[:20],
                title='Coefficients by feature',
                x_label='Coefficient',
                y_label='Features'
            )
    else:
        results_df = get_feature_importance_df(model['reg'], pruned_features)
        factor = 100

        # if plot and results_df is not None:
        #     plot_results(results_df[:20], factor)

        plot_barplot(
            'importance',
            results_df[:20] * factor,
            title='Coefficients by feature',
            x_label='Coefficient',
            y_label='Features'
        )

    plt.show()


def plot_cities_results(model, population_tests_dir, plot=True):
    """
    Retrieves and plots the population comparison results.

    :param model: fitted model
    :param population_tests_dir: directory pointing to where the population tests are
    :param plot: whether results should be plotted or not.
    :return: None
    """

    population_tests, file_names = import_polygons_data(population_tests_dir)

    population_results = {
        "prediction": {},
        "actual": {},
        "difference": {}
    }

    for i in range(population_tests.shape[0]):

        test_case = file_names[i]
        city_name = ' '.join([item[0].upper() + item[1:] for item in test_case.split('-')])

        test_row = population_tests.iloc[i:i+1, :].copy()

        test_row["area"] = test_row["geometry"].apply(lambda poly: calculate_area(poly))

        pred = model.predict(test_row) * test_row["area"]

        population_results["prediction"][city_name] = pred.values[0]
        population_results["actual"][city_name] = tests_actual[test_case] \
            if test_case in tests_actual else None

        try:
            diff = (population_results["prediction"][city_name] - population_results["actual"][city_name]) \
                   / population_results["actual"][city_name]
            population_results["difference"][city_name] = round(diff, 2)

        except TypeError:
            population_results["difference"][test_case] = 'N/A'

    results_df = pd.DataFrame(population_results)

    if plot:
        plot_comparisons(results_df)


def model_evaluation(model, X_test, y_test, features, population_tests_dir, grid_search=False, plot=True):
    """
    Wrapper around the model evaluation steps.

    :param model: fitted model
    :param X_test: test features data frame
    :param y_test: test target data frame
    :param features: original sequence of features
    :param population_tests_dir: directory pointing to where the population tests are
    :param grid_search: whether to perform grid search or not
    :param plot: whether results should be plotted or not.
    :return:
    """

    if grid_search:
        logging.info(f"best estimator: {model.best_estimator_}")
        logging.info(f"best params: {model.best_params_}")
        logging.info(f"best score: {model.best_score_}")

        return

    total_score = model.score(X_test, y_test)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f'Total score: {total_score}')
    logging.info(f'Mean absolute error: {mae}')

    plot_feature_results(model, features, plot)

    plot_cities_results(model, population_tests_dir, plot)
