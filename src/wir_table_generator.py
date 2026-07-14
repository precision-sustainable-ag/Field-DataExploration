import logging
from datetime import datetime, timezone

from omegaconf import DictConfig
from tqdm import tqdm

from db.connection import get_connection
from db.upsert import upsert_image_from_imageref, upsert_sample_attributes
from ingestion_sources import AzureTableSource, resolve_table_credentials
from utils.utils import read_yaml

log = logging.getLogger(__name__)


class TableExporter:
    """
    Pulls Azure Table Storage data, driven by `cfg.sources` (entries with
    `type: azure_table`), and upserts rows into the SQLite DB according to each
    source's `entity` field:
        - image_ref: one row per image, links a blob to a MasterRefID
        - sample_attributes: staged raw rows, coalesced into `samples` later
        - none: no DB write (e.g. wirlogs)

    Attributes:
        __auth_config_data (dict): Azure Table Storage credentials per table.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.__auth_config_data = read_yaml(cfg.pipeline_keys)
        self.db_path = cfg.paths.db_path
        self.sources = [s for s in cfg.sources if s.type == "azure_table"]

    def pull_and_upsert(self):
        conn = get_connection(self.db_path)
        try:
            for source in tqdm(self.sources):
                url, sas_token = resolve_table_credentials(self.__auth_config_data, source.key_ref)
                entities = AzureTableSource(source.name, url, sas_token).fetch()
                if not entities:
                    log.warning(f"{source.name} data is empty, Not saving!")
                    continue

                log.info(f"Fetched {len(entities)} {source.name} rows")
                self._upsert_entities(conn, source, entities)
            conn.commit()
        finally:
            conn.close()

    def _upsert_entities(self, conn, source, entities) -> None:
        if source.entity == "image_ref":
            upserted = sum(upsert_image_from_imageref(conn, entity) for entity in entities)
            skipped = len(entities) - upserted
            log.info(f"{source.name}: upserted {upserted} images, skipped {skipped} rows with no MasterRefID")
        elif source.entity == "sample_attributes":
            ingested_at = datetime.now(timezone.utc).isoformat()
            master_ref_key = source.get("master_ref_key", "MasterRefID")
            upserted = sum(
                upsert_sample_attributes(conn, source.name, entity, ingested_at, master_ref_key)
                for entity in entities
            )
            skipped = len(entities) - upserted
            log.info(f"{source.name}: upserted {upserted} sample attribute rows, skipped {skipped} rows with no {master_ref_key}")
        elif source.entity == "none":
            pass
        else:
            log.warning(f"{source.name}: unrecognized entity type {source.entity!r}, skipping DB upsert")


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    exporter = TableExporter(cfg)
    exporter.pull_and_upsert()
    log.info(f"{cfg.general.task} completed.")
