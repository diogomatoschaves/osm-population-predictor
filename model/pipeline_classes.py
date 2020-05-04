import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

from model.helper_methods import feature_mapping
from preprocessing.process_data import calculate_area


class CalculateArea(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):

        X_copy = X.copy()

        if 'area' not in X_copy.columns:
            X_copy['area'] = X_copy['geometry'].apply(lambda poly: calculate_area(poly))

        return X_copy


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        X_copy = X.copy()

        X_copy = X_copy[self.columns]

        return X_copy


class AreaNormalizer(BaseEstimator, TransformerMixin):

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


class RemoveColumn(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):

        X_copy = X.drop(columns=self.columns)

        return X_copy


class FeaturePolynomial(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):

        poly_X = feature_mapping(X.to_numpy(), 3, only_self_terms=True)

        return poly_X