import logging
import sys


def configure_logger():
    logger = logging.getLogger()

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("shapely").setLevel(logging.ERROR)
    log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(log_handler)

    return logger
