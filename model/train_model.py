import logging

import joblib

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.pipeline_classes import CalculateArea, FeatureSelector, AreaNormalizer, RemoveColumns, FeaturePolynomial
from model.model_evaluation import model_evaluation
from settings import feature_mapping_params, regressor_params, regressor_name, not_features

grid_search_params_defaults = {
    "reg__n_estimators": [250, 300, 350],
    "reg__min_samples_split": [2, 4, 5],
    "reg__max_features": ["sqrt", "log2", "auto"],
    "reg__max_depth": [2, 3, 5, 6],
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
    features, reg_name=None, grid_search=False, grid_search_params=None
):
    """
    Builds the pipeline required to fit the features into the model
    :param features: features to learn from
    :param reg_name: classifier name. Default will be RandomForestRegressor
    :param grid_search: If grid search should be performed
    :param grid_search_params: extra params to be passed to the grid search
    :return: the built model
    """

    if not grid_search_params:
        grid_search_params = {}

    try:
        if not reg_name:
            reg_name = "GradientBoostingRegressor"

        regressor = eval(reg_name)(**regressor_params[reg_name])
    except NameError:
        raise Exception(f"{reg_name} is not a valid Regressor")

    pipeline = Pipeline([
        ('calculate_area', CalculateArea()),
        ('feature_selector', FeatureSelector(columns=features)),
        ('area_normalizer', AreaNormalizer()),
        ('remove_columns', RemoveColumns(['area'])),
        ('feature_mapping', FeaturePolynomial(**feature_mapping_params)),
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


def train_model(df_model, population_tests_dir, grid_search):
    """
    Handles all the logic to build a machine learning pipeline, train the data and
    evaluate the model.

    :param df_model: preprocessed data frame containing the training data
    :param population_tests_dir: directory pointing to where the population tests are
    :param grid_search: whether to perform grid search on the model or not
    :return: the fitted model
    """

    grid_search_params = {
        "feature_mapping__order": [1, 2, 3],
        "reg": [Ridge(max_iter=2000), Lasso(max_iter=2000)],
        "reg__alpha": [3, 5, 10, 50, 100],
    }

    features_vars = [col for col in df_model.columns if col not in not_features]
    target_var = 'population'

    X = df_model
    y = df_model[target_var] / df_model['area']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    logging.info("\tbuilding model...")

    model = build_model(
        features_vars,
        grid_search=grid_search,
        grid_search_params=grid_search_params,
        reg_name=regressor_name
    )

    if grid_search:
        logging.info('\tPerforming grid search...')
    else:
        logging.info("\tfitting data...")

    model.fit(X_train, y_train)

    logging.info("\tevaluating model...")

    model_evaluation(model, X_test, y_test, features_vars, population_tests_dir, grid_search)

    return model
