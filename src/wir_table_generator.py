import logging
from pathlib import Path

import pandas as pd
from azure.core.credentials import AzureSasCredential
from azure.data.tables import TableServiceClient
from omegaconf import DictConfig
from tqdm import tqdm

from utils.utils import read_yaml

log = logging.getLogger(__name__)


class TableExporter:
    """
    A class for exporting Azure Table Storage data to CSV files.

    This class handles the connection to Azure Table Storage, retrieves data from specified tables,
    and exports the data into CSV files within a specified directory.

    Attributes:
        __auth_config_data (dict): Configuration data containing Azure Table Storage credentials,
                                   including account URLs and SAS tokens for each table.
        tables_dir (str): The directory path where CSV files will be stored.

    Methods:
        __init__(cfg: DictConfig): Initializes the TableExporter instance with configuration from a DictConfig object.
        get_table_data(account_url, sas_token, table_name): Retrieves data from a specified table in Azure Table Storage.
        get_table_csv(): Iterates through configured tables, retrieves their data, and exports the data to CSV files.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.__auth_config_data = read_yaml(cfg.pipeline_keys)
        self.tables_dir = cfg.paths.tablesdir
        Path(self.tables_dir).mkdir(exist_ok=True, parents=True)

    def get_table_data(self, account_url, sas_token, table_name):

        try:
            table_service_client = TableServiceClient(
                endpoint=account_url, credential=AzureSasCredential(sas_token)
            )
            table_client = table_service_client.get_table_client(table_name=table_name)
            # Fetch all entities from the specified table
            entities = []
            for i in table_client.list_entities():
                timestamp = str(i._metadata["timestamp"])
                i["Timestamp"] = timestamp
                entities.append(i)
            return entities

        except Exception as error:
            log.exception(f"Error! Check {table_name} authorization parameters")
            return []

    def get_table_csv(self):
        for table_name in tqdm(self.__auth_config_data["tables"]):
            sas_token = self.__auth_config_data["tables"][table_name]["sas_token"]
            account_url = self.__auth_config_data["tables"][table_name]["url"]
            # Get data from Azure Table Storage
            table_data = self.get_table_data(account_url, sas_token, table_name)
            if table_data:
                df_images_details = pd.DataFrame(table_data)
                # Export to CSV
                csv_path = Path(self.tables_dir, f"{table_name}_table_metrics.csv")
                df_images_details.to_csv(csv_path, index=False)
                log.info(f"Exported {table_name} data to {csv_path}")
            else:
                log.warn(f"{table_name} data is empty, Not saving!")


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    exporter = TableExporter(cfg)
    exporter.get_table_csv()
    log.info(f"{cfg.general.task} completed.")
