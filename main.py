import sys
import os
import logging

from model.train_model import train_model, save_model
from preprocessing.process_data import process_data

try:
    directory_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(directory_path)
except NameError:
    pass

from utils.logger import configure_logger
from utils.config_parser import get_config
from settings import grid_search


def main():

    # ============================ Setup ===================================

    configure_logger()

    config = get_config()

    # ========================== Load & prepare input data ==============================

    logging.info("Processing base data...")

    base_data_df = process_data(
        base_data_dir=config.base_data_dir, input_data_file=config.input_data_file
    )

    # ========================= Train model ===============================

    logging.info("Training model and evaluating...")

    model = train_model(base_data_df, config.population_tests_dir, grid_search)

    # ========================== Export Results ===================================

    logging.info("Saving model...")

    save_model(model, config.out_file)


if __name__ == "__main__":
    main()
