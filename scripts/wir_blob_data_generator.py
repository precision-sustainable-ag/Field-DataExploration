#!/usr/bin/env python3
import logging
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient
from omegaconf import DictConfig
from tqdm import tqdm
from utils.utils import read_yaml

log = logging.getLogger(__name__)


class BlobMetricExporter:
    """
    A class designed to export metrics of blob files from Azure Blob Storage to CSV files.

    This class connects to Azure Blob Storage, retrieves detailed metrics for files within specified containers,
    and exports these metrics into CSV files for further analysis or reporting.

    Attributes:
        __auth_config_data (dict): Configuration data containing Azure Blob Storage credentials,
                                including account URLs and SAS tokens for each container.
        blobs_dir (Path): The directory path where CSV files will be stored.

    Methods:
        __init__(cfg): Initializes the BlobMetricExporter instance with configuration from a DictConfig object.
        get_blob_metrics(account_url, sas_token, container_name): Retrieves file metrics from a specified container in Azure Blob Storage.
        get_blob_csv(): Iterates through configured containers, retrieves their file metrics, and exports the metrics to CSV files.
    """

    def __init__(self, cfg) -> None:
        self.__auth_config_data = read_yaml(cfg.pipeline_keys)
        self.blobs_dir = cfg.data.blobsdir
        Path(self.blobs_dir).mkdir(exist_ok=True, parents=True)

    def get_blob_metrics(self, account_url, sas_token, container_name):
        """
        Uses container client to return detailed metrics for jpg and raw files in the container.

        Returns:
            images_details (list): List of dictionaries with details for each image.
        """
        try:

            images_details = []

            blob_service_client = BlobServiceClient(
                account_url=account_url, credential=sas_token
            )
            container_client = blob_service_client.get_container_client(container_name)

            for blob in tqdm(container_client.list_blobs()):

                image_detail = {
                    "name": blob.name,  # blob name
                    "memory_mb": float(blob.size / pow(1024, 2)),  # convert kb to mb
                    "container": blob.container,  # get container name
                    "creation_time_utc": blob.creation_time,  # get creation time
                }
                images_details.append(image_detail)

            return images_details

        except Exception as error:

            log.exception(f"Error! Check {container_name} authorization parameters")

    def get_blob_csv(self):
        for container_name in tqdm(self.__auth_config_data["blobs"]):
            sas_token = self.__auth_config_data["blobs"][container_name]["sas_token"]
            account_url = self.__auth_config_data["blobs"][container_name]["url"]

            # Get data from Blob servers
            images_details = self.get_blob_metrics(
                account_url, sas_token, container_name
            )
            if images_details:
                df_images_details = pd.DataFrame(images_details)
                # Export to CSV
                csv_path = Path(self.blobs_dir, f"{container_name}_blob_metrics.csv")
                df_images_details.to_csv(csv_path, index=False)
                log.info(f"Exported {container_name} data to {csv_path}")
            else:
                log.warn(f"{container_name} data is empty, Not saving!")


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    exporter = BlobMetricExporter(cfg)
    exporter.get_blob_csv()
    log.info(f"{cfg.general.task} completed.")
