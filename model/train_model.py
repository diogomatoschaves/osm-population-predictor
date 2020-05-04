import logging

import joblib

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_evaluation.plot import grid_search as grid_plot

from model.pipeline_classes import CalculateArea, FeatureSelector, AreaNormalizer, RemoveColumn
from model.tests_data import tests_areas, tests_actual
from preprocessing.process_data import import_polygons_data

regressor_params_defaults = {
    "RandomForestRegressor": {
        "n_estimators": 200
    },
    "GradientBoostingRegressor": {
        "n_estimators": 300,
        "max_depth": 5, "max_features": 'sqrt',
        "min_samples_split": 5
    },
}

grid_search_params_defaults = {
    "reg__n_estimators": [250, 300, 350],
    "reg__min_samples_split": [2, 4, 5],
    "reg__max_features": ["sqrt", "log2", "auto"],
    "reg__max_depth": [2, 3, 5, 6],
    # "reg__min_samples_leaf": [1, 2, 3]
}


def save_model(model, model_filepath):
    """
    Saves the model as a pickle file
    :param model: the ML model to be saved
    :param model_filepath: path pointing to where the file should be saved
    :return: None
    """

    joblib.dump(model, model_filepath)


def load_model(model_filepath):
    """
    Loads the model from a pickle file

    :param model_filepath: path pointing to the saved model
    :return: the loaded model
    """

    model = joblib.load(model_filepath)

    return model


def build_model(
    features, reg_name=None, grid_search=False, regressor_params=None, grid_search_params=None
):
    """
    Builds the pipeline required to fit the features into the model
    :param features: features to learn from
    :param reg_name: classifier name. Default will be RandomForestRegressor
    :param grid_search: If grid search should be performed
    :param regressor_params: extra params to be passed to the classifier
    :param grid_search_params: extra params to be passed to the grid search
    :return: the built model
    """

    if not regressor_params:
        regressor_params = {}

    if not grid_search_params:
        grid_search_params = {}

    try:

        if not reg_name:
            reg_name = "GradientBoostingRegressor"
        if len(regressor_params) == 0:
            regressor_params.update(regressor_params_defaults[reg_name])

        regressor = eval(reg_name)(**regressor_params)
    except NameError:
        raise Exception(f"{reg_name} is not a valid Regressor")

    pipeline = Pipeline([
        ('calculate_area', CalculateArea()),
        ('feature_selector', FeatureSelector(columns=features)),
        ('area_normalizer', AreaNormalizer()),
        ('remove_columns', RemoveColumn(['area'])),
        ('pre-processing', StandardScaler()),
        ('reg', regressor)
    ])

    if grid_search:

        param_grid = {}

        if len(grid_search_params) == 0:
            param_grid.update(grid_search_params_defaults)
        else:
            param_grid.update(grid_search_params)

        pipeline = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1, cv=3)

    return pipeline


def get_feature_importance_df(model, features):
    importance_df = pd.DataFrame(index=[col.replace('_', ' ') for col in features])
    importance_df['importance'] = model.feature_importances_
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df


def get_coefficients_df(reg, features):

    coefs_df = pd.DataFrame(index=[col.replace('_', ' ') for col in features])
    coefs_df['coefs'] = reg.coef_
    coefs_df['abs_coefs'] = np.abs(reg.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)

    return coefs_df


def model_evaluation(model, X_test, y_test, features, population_tests_dir, grid_search=False, plot=True):

    if grid_search:
        logging.info(f"best estimator: {model.best_estimator_}")
        logging.info(f"best params: {model.best_params_}")
        logging.info(f"best score: {model.best_score_}")
        logging.info(f"all_results: {model.cv_results_}")

        return

    pruned_features = [feature for feature in features if feature != 'area']

    if isinstance(model['reg'], (LinearRegression, Ridge, Lasso)):
        results_df = get_coefficients_df(model['reg'], pruned_features)
    else:
        results_df = get_feature_importance_df(model['reg'], pruned_features)

    if plot:
        plot_results(results_df[:20])

    total_score = model.score(X_test, y_test)

    logging.info(f'Total score: {total_score}')

    population_tests, file_names = import_polygons_data(population_tests_dir)

    population_results = {
        "prediction": {},
        "actual": {}
    }

    for i in range(population_tests.shape[0]):

        test_case = file_names[i]

        test_row = population_tests.iloc[i:i+1, :].copy()

        test_row["area"] = tests_areas[test_case]

        pred = model.predict(test_row) * test_row["area"]

        population_results["prediction"][test_case] = pred.values[0]
        population_results["actual"][test_case] = tests_actual[test_case]

    results_df = pd.DataFrame(population_results)

    if plot:
        plot_results(results_df, factor=1E-6)


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
        cbar_kws={'format': '%.0f%%'}
    )

    plt.show()


def train_model(df_model, population_tests_dir, grid_search):

    not_features = [
        'updated',
        'ADMIN',
        'ISO_A3',
        'building_count',
        'count',
        'osm_users',
        'geometry',
        'gdp',
        'population',
        'area_km2',
        'highway_sum',
        'highway_length',
        'osm_objects'
    ]

    features_vars = [col for col in df_model.columns if col not in not_features]
    target_var = 'population'

    X = df_model
    y = df_model[target_var] / df_model['area']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logging.info("\tbuilding model...")

    model = build_model(features_vars, grid_search=grid_search)

    if grid_search:
        logging.info('\tPerforming grid search...')
    else:
        logging.info("\tfitting data...")

    model.fit(X_train, y_train)

    logging.info("\tevaluating model...")

    model_evaluation(model, X_test, y_test, features_vars, population_tests_dir, grid_search)

    return model
