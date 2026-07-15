import logging

from omegaconf import DictConfig

from db.connection import get_connection
from db.upsert import upsert_image_from_blob
from ingestion_sources import AzureBlobSource, resolve_blob_credentials
from utils.utils import read_yaml

log = logging.getLogger(__name__)


class BlobMetricExporter:
    """
    Pulls Azure Blob Storage metrics, driven by `cfg.sources` (entries with
    `type: azure_blob`), and upserts each blob into the `images` table in the
    SQLite DB.

    Attributes:
        __auth_config_data (dict): Azure Blob Storage credentials per container.
    """

    def __init__(self, cfg) -> None:
        self.__auth_config_data = read_yaml(cfg.pipeline_keys)
        self.db_path = cfg.paths.db_path
        self.sources = [s for s in cfg.sources if s.type == "azure_blob"]

    def pull_and_upsert(self):
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

                log.info(f"Fetched {len(blob_details)} {source.name} blobs")

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
    exporter.pull_and_upsert()
    log.info(f"{cfg.general.task} completed.")
