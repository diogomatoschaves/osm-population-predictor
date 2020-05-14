import logging
import os

import boto3

from utils.config_parser import get_config
from utils.logger import configure_logger


def download_dir(client, resource, dist, local, bucket):

    paginator = client.get_paginator('list_objects_v2')

    page_iterator = paginator.paginate(Bucket=bucket)

    for result in page_iterator:

        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket)

        number_files = len(result.get('Contents', []))

        for i, file in enumerate(result.get('Contents', [])):

            file_name = file.get('Key')

            logging.info(f'\tDownloading {file_name} ({i + 1}/{number_files})...')

            dest_pathname = os.path.join(local, file_name)

            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            resource.meta.client.download_file(bucket, file.get('Key'), dest_pathname)


def download(base_data_dir, bucket):

    logging.info(f'Downloading files and saving in {base_data_dir}/...')

    client = boto3.client('s3')
    resource = boto3.resource('s3')
    download_dir(client, resource, '', local=base_data_dir, bucket=bucket)


if __name__ == "__main__":

    boto3.set_stream_logger('', logging.ERROR)

    configure_logger(("boto3", "CRITICAL"))

    config = get_config()

    download(config.base_data_dir, config.bucket)

