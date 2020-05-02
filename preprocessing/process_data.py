import logging
import os
import re

import pandas as pd
import geopandas as gpd
from turf import polygon, feature_collection, area
from turf.utils.exceptions import InvalidInput


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

        file_names.append(filename.split('_')[0])

        if df is not None:
            df = pd.concat([df, gpd.read_file(file_path)])
        else:
            df = gpd.read_file(file_path)

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

    df = df[df[filter_column]]

    return df


def delete_empty_columns(df):

    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(columns=col)

    return df


def get_regex_matches(regex_str):

    pattern = r"\((.*?)\)"

    matches = re.findall(pattern, regex_str)

    matches = [match.replace("(", "").replace(")", "").split(", ") for match in matches]

    return matches


def handle_multi_polygon(multi_polygon):

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

    if poly.geom_type == 'MultiPolygon':
        coords = handle_multi_polygon(poly.wkt)
    else:
        coords = [list(coord) for coord in poly.exterior.coords]

    a = area([coords]) / 1E6

    return a


def process_data(base_data_dir, input_data_file):

    logging.info("\tImporting data...")

    base_data_df = load_data(base_data_dir, input_data_file)

    logging.info("\tCleaning data...")

    base_data_df = filter_data(base_data_df, "updated")

    base_data_df = delete_empty_columns(base_data_df)

    base_data_df['area'] = base_data_df['geometry'].apply(lambda poly: calculate_area(poly))

    logging.info(f'Data has {base_data_df.shape[0]} rows and {base_data_df.shape[1]} columns')

    return base_data_df
