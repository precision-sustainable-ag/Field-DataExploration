#!/usr/bin/env python3
# fmt: off
# isort: off
import logging

import pandas as pd
from omegaconf import DictConfig

from create_batches import CreateBatchProcessor, FieldBatchLister
from db.connection import get_connection
from db.locations import all_known_locations, batch_folder_regex
from utils.utils import read_yaml

log = logging.getLogger(__name__)

"""
    create_batches.py's batching pipeline (preprocess -> group into 3-hourly
    sub-batches -> assign batch folders), sourced from images/samples/locations
    in the DB instead of the CSV + find_most_recent_csv. Not part of the
    automatic pipeline (cfg.pipeline) yet - run manually and diff its batch-folder
    assignments against the legacy CreateBatchProcessor's:
        python main.py general.task=create_batches_db +pipeline=[create_batches_db]
"""

BATCH_SOURCE_QUERY = """
    SELECT
        images.blob_name AS Name,
        images.base_name AS BaseName,
        images.extension AS Extension,
        images.exif_datetime AS CameraInfo_DateTime,
        images.has_matching_jpg_and_raw AS HasMatchingJpgAndRaw,
        images.master_ref_id AS MasterRefID,
        samples.location_code AS UsState
    FROM images
    LEFT JOIN samples ON images.master_ref_id = samples.master_ref_id
"""


class DbBatchProcessor(CreateBatchProcessor):
    """Same batching logic as CreateBatchProcessor (split_datetime, preprocess_df,
    adjust_groups, filter_batched_data, ...), sourced from the DB instead of a CSV."""

    def __init__(self, cfg: DictConfig) -> None:
        self.ykeys = read_yaml(cfg.pipeline_keys)
        self.file_path = "./tempoutputfieldbatches.txt"
        self.conn = get_connection(cfg.paths.db_path)
        self.read_and_convert_datetime()

    def read_and_convert_datetime(self) -> None:
        """Loads batch source data from images/samples. Datetimes are already
        normalized at ingest (db/normalize.py), so unlike the CSV-driven parent,
        no ':'->'-' regex pass is needed here."""
        self.df = pd.read_sql_query(BATCH_SOURCE_QUERY, self.conn)
        self.df["CameraInfo_DateTime"] = pd.to_datetime(
            self.df["CameraInfo_DateTime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

    def warn_on_unknown_batch_labels(self) -> None:
        """Flags any synthesized batch folder whose location prefix isn't a known
        location code, instead of the legacy behavior of such folders being
        silently skipped later with no error (plan section 3.4)."""
        locations = all_known_locations(self.conn)
        regex = batch_folder_regex(locations)
        labels = self.df["batches"].str.split("/raws/").str[0].unique()
        unknown = sorted(label for label in labels if not regex.match(label))
        if unknown:
            log.warning(
                f"{len(unknown)} synthesized batch labels don't match any known "
                f"location code: {unknown[:10]}{'...' if len(unknown) > 10 else ''}"
            )


def main(cfg: DictConfig) -> None:
    """Mirrors create_batches.py's main(), sourcing the batch DataFrame from the
    DB instead of the CSV."""
    log.info(f"Starting {cfg.general.task}")

    present_batches_df = FieldBatchLister(cfg).df
    present_batches_df.to_csv("present_batches.csv", index=False)

    dataproc = DbBatchProcessor(cfg)
    try:
        dataproc.config_keys()
        dataproc.split_datetime()
        dataproc.preprocess_df()
        dataproc.adjust_groups()
        dataproc.warn_on_unknown_batch_labels()
        dataproc.filter_batched_data(present_batches_df)

        run_concurrent = True
        if run_concurrent:
            dataproc.process_df_concurrently()
        else:
            dataproc.process_df()
    finally:
        dataproc.conn.close()

    log.info(f"Task '{cfg.general.task}' completed successfully.")
