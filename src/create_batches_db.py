#!/usr/bin/env python3
# fmt: off
# isort: off
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobServiceClient
from omegaconf import DictConfig
from tqdm import tqdm

from db.connection import get_connection
from db.locations import all_known_locations, batch_folder_regex
from db.upsert import update_image_batch_id, upsert_batches
from utils.utils import read_yaml

log = logging.getLogger(__name__)

"""
    create_batches.py's batching pipeline (preprocess -> group into 3-hourly
    sub-batches -> assign batch folders), sourced from images/samples/locations
    in the DB instead of the CSV + find_most_recent_csv. Not part of the
    automatic pipeline (cfg.pipeline) yet - deliberately kept commented out
    there, since it moves real blobs via azcopy. Run manually:
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


class FieldBatchLister:
    """Lists what's already been moved into the `field-batches` container, so
    filter_batched_data() knows what not to move again."""

    def __init__(self, cfg):
        self.keys = self.config_keys(cfg.pipeline_keys)
        self.timestamp = datetime.now().strftime("%Y%m%d")
        all_blobs = self.list_unique_folders()
        self.df = self.get_blobs_per_batch(all_blobs)

    def config_keys(self, keypath):
        yamkeys = read_yaml(keypath)
        self.field_batches_sas_token = yamkeys["blobs"]["field-batches"]["read_sas_token"]
        self.field_batches_url = yamkeys["blobs"]["account_url"]
        self.container_name = "field-batches"
        log.debug("WeedsImageRepo keys configured.")

    def list_unique_folders(self):
        blob_service_client = BlobServiceClient(account_url=self.field_batches_url, credential=self.field_batches_sas_token)
        container_client = blob_service_client.get_container_client(self.container_name)
        all_blobs = []
        for blob in tqdm(container_client.list_blobs()):
            blob_name = blob.name
            if ("raws" in blob_name) and not ("preprocessed" in blob_name):
                all_blobs.append(blob_name)
        return all_blobs

    def get_blobs_per_batch(self, all_blobs):
        df = pd.DataFrame(all_blobs, columns=["BlobName"])
        df[["BatchID", "Subfolder1", "Subfolder2", "FileName"]] = df["BlobName"].str.split("/", n=4, expand=True)
        df["BaseName"] = df["FileName"].str.rsplit(".", n=1).str[0]
        df = df.sort_values(by=["BatchID"])
        df["BatchFolder"] = df["BatchID"] + "/" + df["Subfolder1"] + "/" + df["Subfolder2"]
        return df


def round_down_to_nearest_3_hours(dt: datetime) -> datetime:
    rounded_hour = dt.hour - (dt.hour % 3)
    return dt.replace(hour=rounded_hour, minute=0, second=0, microsecond=0)


def jpg_name_to_arw(jpg_name: str) -> str:
    """Derives the corresponding RAW filename for a JPG blob name, matching '.jpg'
    case-insensitively (source filenames are inconsistently '.JPG'/'.jpg')."""
    return re.sub(r"\.jpg$", ".ARW", jpg_name, flags=re.IGNORECASE)


class DbBatchProcessor:
    """create_batches.py's CreateBatchProcessor, sourced from images/samples in
    the DB instead of the CSV + find_most_recent_csv."""

    def __init__(self, cfg: DictConfig) -> None:
        self.ykeys = read_yaml(cfg.pipeline_keys)
        self.file_path = "./tempoutputfieldbatches.txt"
        self.conn = get_connection(cfg.paths.db_path)
        self.read_and_convert_datetime()

    def config_keys(self) -> None:
        self.write_fbatch_key = self.ykeys["blobs"]["field-batches"]["write_sas_token"]
        self.fbatch_url = self.ykeys["blobs"]["field-batches"]["url"]
        self.read_weedimgrepo_key = self.ykeys["blobs"]["weedsimagerepo"]["sas_token"]
        self.weedimgrepo_url = self.ykeys["blobs"]["weedsimagerepo"]["url"]
        log.debug("Configured keys for image repository and field batches.")

    def read_and_convert_datetime(self) -> None:
        """Loads batch source data from images/samples. Datetimes are already
        normalized at ingest (db/normalize.py), so unlike the CSV-driven original,
        no ':'->'-' regex pass is needed here."""
        self.df = pd.read_sql_query(BATCH_SOURCE_QUERY, self.conn)
        self.df["CameraInfo_DateTime"] = pd.to_datetime(
            self.df["CameraInfo_DateTime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
        )

    def split_datetime(self) -> "DbBatchProcessor":
        self.df["CameraInfo_Date"] = self.df["CameraInfo_DateTime"].dt.date
        self.df["CameraInfo_Time"] = self.df["CameraInfo_DateTime"].dt.time
        return self

    def preprocess_df(self) -> "DbBatchProcessor":
        log.info("Preprocessing DataFrame")
        self.df = self.df[self.df["HasMatchingJpgAndRaw"] == True].dropna(subset=["UsState"])
        self.df = self.df[self.df["Extension"] == "jpg"]
        self.df = self.df.dropna(subset="CameraInfo_DateTime")
        return self

    def adjust_groups(self) -> "DbBatchProcessor":
        """Groups images into 3-hourly sub-batches per (UsState, CameraInfo_Date)
        and assigns each a batch folder path."""
        log.info("Adjusting groups for batch processing")
        self.df = self.df.sort_values(by=["UsState", "MasterRefID", "CameraInfo_DateTime"])
        self.df["ThreeHourlyGroup"] = self.df["CameraInfo_DateTime"].apply(round_down_to_nearest_3_hours)
        self.df["SubBatchIndex"] = self.df.sort_values(by=["ThreeHourlyGroup"]).groupby(["UsState", "CameraInfo_Date"])["ThreeHourlyGroup"].transform(lambda x: pd.factorize(x)[0] + 1)
        self.df["SubBatchIndex_Padded"] = self.df["SubBatchIndex"].apply(lambda x: f"{x:0{2}d}")
        self.df = self.df.sort_values(by=["UsState", "CameraInfo_Date", "SubBatchIndex"])
        self.df["RawName"] = self.df["Name"].apply(jpg_name_to_arw)
        self.df["batches"] = self.df.apply(lambda row: f"{row['UsState']}_{row['CameraInfo_Date'].strftime('%Y-%m-%d')}/raws/{row['SubBatchIndex_Padded']}/{row['RawName']}", axis=1)
        return self

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

    def persist_batches(self) -> None:
        """Upserts a `batches` row per assigned batch and sets images.batch_id,
        reusing upsert_batches's label-parsing logic (db/upsert.py, shared with
        migrate_to_db.py) instead of duplicating it. Runs over the full computed
        assignment (self.df, before filter_batched_data narrows it to "not yet
        moved"), so images.batch_id reflects batch membership regardless of
        whether the azcopy copy has happened yet."""
        self.df["BatchID"] = self.df["UsState"] + "_" + self.df["CameraInfo_Date"].apply(lambda d: d.strftime("%Y-%m-%d"))
        label_to_id = upsert_batches(self.conn, self.df)
        batch_id_by_blob_name = {
            row.Name: label_to_id[row.BatchID]
            for row in self.df.itertuples(index=False)
            if row.BatchID in label_to_id
        }
        update_image_batch_id(self.conn, batch_id_by_blob_name)
        self.conn.commit()
        log.info(f"Persisted {len(label_to_id)} batches, set batch_id on {len(batch_id_by_blob_name)} images")

    def filter_batched_data(self, present_batches_df) -> None:
        """Filters out already processed batches and raises on duplicate image names."""
        self.df = self.df[~self.df["BaseName"].isin(present_batches_df["BaseName"])]
        if len(self.df) == 0:
            log.info("No new images present. No images to be moved to the field-batches blob containers. Exiting.")
            exit(0)

        self.df[["BatchID_y", "Subfolder1", "Subfolder2", "FName"]] = self.df["batches"].str.split("/", expand=True)
        self.df["BatchFolder"] = self.df["BatchID_y"] + "/" + self.df["Subfolder1"] + "/" + self.df["Subfolder2"]

        duplicate_image_names = self.df[self.df.duplicated(subset=["Name"], keep=False)]
        if duplicate_image_names.empty:
            log.info("No duplicates")
        else:
            log.error("Duplicates found in batch folders. Saving duplicates to 'duplicate_image_names.csv'")
            duplicate_image_names.to_csv("duplicate_image_names.csv", index=False)
            raise ValueError("Duplicates image names found in batch folders. Please resolve before proceeding.")

    def move_from_weeedsimagerepo2fieldbatches(self, batch: str) -> None:
        """Moves a batch from the weeds image repository to field batches using azcopy."""
        log.info(f"Moving batch {batch} from weeds image repository to field batches")
        blob_name = Path(batch).name
        weedimgrepo_src = f"{self.weedimgrepo_url}/{blob_name}{self.read_weedimgrepo_key}"
        fieldbatch_dst = f"{self.fbatch_url}/{batch}{self.write_fbatch_key}"

        command = f'azcopy copy "{weedimgrepo_src}" "{fieldbatch_dst}" --recursive'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            log.info("Copy successful")
            log.debug(result.stdout)
        else:
            log.error("Error in copy operation")
            log.debug(result.stderr)

    def process_df(self) -> None:
        log.info("Processing DataFrame without concurrency")
        for _, row in self.df.reset_index().iterrows():
            self.move_from_weeedsimagerepo2fieldbatches(row["batches"])

    def process_df_concurrently(self) -> None:
        log.info("Processing DataFrame with concurrency")
        batches = self.df["batches"].unique()
        cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        max_workers = max(1, int(cpu_count / 3))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.move_from_weeedsimagerepo2fieldbatches, batch) for batch in batches]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Error processing batch: {e}")


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
        dataproc.persist_batches()
        dataproc.filter_batched_data(present_batches_df)

        run_concurrent = True
        if run_concurrent:
            dataproc.process_df_concurrently()
        else:
            dataproc.process_df()
    finally:
        dataproc.conn.close()

    log.info(f"Task '{cfg.general.task}' completed successfully.")
