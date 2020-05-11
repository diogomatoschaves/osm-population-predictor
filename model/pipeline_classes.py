import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from model.helper_methods import feature_mapping
from preprocessing.process_data import calculate_area


class CalculateArea(BaseEstimator, TransformerMixin):

    """
    Calculates area of entry if it's not present in features already.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):

        X_copy = X.copy()

        if 'area' not in X_copy.columns:
            X_copy['area'] = X_copy['geometry'].apply(lambda poly: calculate_area(poly))

        return X_copy


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects features to be included in the final model.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        X_copy = X.copy()

        X_copy = X_copy[self.columns]

        return X_copy


class AreaNormalizer(BaseEstimator, TransformerMixin):
    """
    Normalizes all features' values by the area.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        X_copy = X.copy()

        if 'area' in X:
            for col in X_copy:
                if col == 'area':
                    continue

                X_copy[col] = X_copy[col] / X_copy['area']

        return pd.DataFrame(X_copy)


class RemoveColumns(BaseEstimator, TransformerMixin):
    """
    Removes selected columns from the model.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):

        X_copy = X.drop(columns=self.columns)

        return X_copy


class FeaturePolynomial(BaseEstimator, TransformerMixin):
    """
    Returns the polynomial terms of features up to a specified degree.
    """

    def __init__(self, order, only_self_terms=True):
        self.order = order
        self.only_self_terms = only_self_terms

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):

        poly_X = feature_mapping(X.to_numpy(), self.order, only_self_terms=self.only_self_terms)

        return poly_X
