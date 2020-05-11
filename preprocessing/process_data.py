import logging
import os
import re

import pandas as pd
import geopandas as gpd
from fiona.errors import DriverError
from turf import polygon, feature_collection, area
from turf.utils.exceptions import InvalidInput

from settings import not_features, delete_outliers_bool


def import_polygons_data(dataset_dir):
    """
    Helper method to load files from a directory and concatenate them
    into a single GeoDataFrame

    :param dataset_dir: path to directory where files are located
    :return: The concatenated GeoDataFrame
    """
    df = None
    file_names = []
    for filename in os.listdir(dataset_dir):

        file_path = os.path.join(dataset_dir, filename)

        try:
            if df is not None:
                df = pd.concat([df, gpd.read_file(file_path)])
            else:
                df = gpd.read_file(file_path)

            file_names.append(filename.split('_')[0])
        except DriverError:
            continue

    return df, file_names


def load_data(base_data_dir, file_name):
    """
    Loads base data from file if it exists, otherwise loads a polygon
    from template.geojson

    :param base_data_dir: path to base data directory
    :param file_name: name of file with base data
    :return: base data data frame, if input file was found or not
    """

    file_path = os.path.join(base_data_dir, file_name)

    base_data_df = gpd.read_file(file_path)

    return base_data_df


def filter_data(df, filter_column):
    """
    Filters a data frame by a boolean column

    :param df: data frame
    :param filter_column: boolean column to filter the data frame by
    :return: the filtered data frame
    """

    df = df[df[filter_column]]

    return df


def delete_empty_columns(df):
    """
    Deletes columns that have only one unique value

    :param df: data frame
    :return: the pruned data frame
    """

    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(columns=col)

    return df


def get_regex_matches(regex_str):
    """
    Finds the matches according to a regex pattern

    :param regex_str: the string to be matched
    :return: the found matches
    """

    pattern = r"\((.*?)\)"

    matches = re.findall(pattern, regex_str)

    matches = [match.replace("(", "").replace(")", "").split(", ") for match in matches]

    return matches


def handle_multi_polygon(multi_polygon):
    """
    Extracts individual polygons from a MultiPolygon shapely representation

    :param multi_polygon: a MultiPolygon shapely representation
    :return: a feature collection of the extracted polygons
    """

    matches = get_regex_matches(multi_polygon)

    polygons = []

    for match in matches:

        coords = [[[float(c) for c in coord.split(" ")] for coord in match]]

        try:
            polygons.append(polygon(coords))
        except InvalidInput:
            coords[0].append(coords[0][0])
            polygons.append(polygon(coords))

    return feature_collection(polygons)


def calculate_area(poly):
    """
    Calculates the area of a polygon in km2
    :param poly: Polygon or MultiPolygon shapely object
    :return: the area of the object
    """

    if poly.geom_type == 'MultiPolygon':
        coords = handle_multi_polygon(poly.wkt)
    else:
        coords = [list(coord) for coord in poly.exterior.coords]

    a = area([coords]) / 1E6

    return a


def detect_outliers(column, df):
    """
    Detects outliers according to a specified rule

    :param column: name of column to detect the outliers on
    :param df: data frame
    :return: the outliers as a data frame
    """

    min_value = df[column].quantile(q=0.0001)
    max_value = df[column].quantile(q=0.9999)

    try:
        outliers = df[(df[column] > max_value) | (df[column] < min_value)]
    except IndexError:
        outliers = None

    return outliers


def delete_outliers(features, df):
    """
    Deletes the outliers of a collection of features

    :param features: sequence of features to delete the outliers on
    :param df: data frame
    :return: the pruned data frame
    """

    outliers = set()
    for feature in features:

        new_outliers = detect_outliers(feature, df)

        if new_outliers is None:
            continue

        outliers.update(new_outliers.index)

    return df[~df.index.isin(outliers)].copy()


def process_data(base_data_dir, input_data_file):
    """
    Handles all the required preprocessing of the data before it can be fed into
    the machine learning pipeline.

    :param base_data_dir: directory of where the base data lives
    :param input_data_file: name of the file containing the input data
    :return: the processed data
    """

    logging.info("\tImporting data...")

    base_data_df = load_data(base_data_dir, input_data_file)

    logging.info("\tCleaning data...")

    base_data_df = filter_data(base_data_df, "updated")

    features = [col for col in base_data_df.columns if col not in not_features]

    if delete_outliers_bool:
        base_data_df = delete_outliers(features, base_data_df)

    base_data_df = delete_empty_columns(base_data_df)

    base_data_df['area'] = base_data_df['geometry'].apply(lambda poly: calculate_area(poly))

    logging.info(f'Data has {base_data_df.shape[0]} rows and {base_data_df.shape[1]} columns')

    return base_data_df

