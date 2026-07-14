import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from db.connection import get_connection
from db.upsert import upsert_image_from_blob
from ingestion_sources import AzureBlobSource, resolve_blob_credentials
from utils.utils import read_yaml

log = logging.getLogger(__name__)


class BlobMetricExporter:
    """
    Exports Azure Blob Storage metrics to CSV files, driven by `cfg.sources`
    (entries with `type: azure_blob`), and upserts each blob into the
    `images` table in the SQLite DB.

    Attributes:
        __auth_config_data (dict): Azure Blob Storage credentials per container.
        blobs_dir (Path): The directory path where CSV files will be stored.
    """

    def __init__(self, cfg) -> None:
        self.__auth_config_data = read_yaml(cfg.pipeline_keys)
        self.blobs_dir = cfg.paths.blobsdir
        self.db_path = cfg.paths.db_path
        self.sources = [s for s in cfg.sources if s.type == "azure_blob"]
        Path(self.blobs_dir).mkdir(exist_ok=True, parents=True)

    def get_blob_csv(self):
        conn = get_connection(self.db_path)
        try:
            for source in self.sources:
                account_url, sas_token, container_name = resolve_blob_credentials(
                    self.__auth_config_data, source.key_ref
                )
                blob_details = AzureBlobSource(source.name, account_url, sas_token, container_name).fetch()
                if not blob_details:
                    log.warning(f"{source.name} data is empty, Not saving!")
                    continue

                df_blob_details = pd.DataFrame(blob_details)
                csv_path = Path(self.blobs_dir, f"{source.name}_blob_metrics.csv")
                df_blob_details.to_csv(csv_path, index=False)
                log.info(f"Exported {source.name} data to {csv_path}")

                if source.entity == "image":
                    for blob in blob_details:
                        upsert_image_from_blob(conn, blob)
                elif source.entity != "none":
                    log.warning(f"{source.name}: unrecognized entity type {source.entity!r}, skipping DB upsert")
            conn.commit()
        finally:
            conn.close()


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    exporter = BlobMetricExporter(cfg)
    exporter.get_blob_csv()
    log.info(f"{cfg.general.task} completed.")
