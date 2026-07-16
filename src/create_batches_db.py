#!/usr/bin/env python3
# fmt: off
# isort: off
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from db.connection import get_connection
from db.locations import all_known_locations, batch_folder_regex
from db.upsert import update_image_batch_id, upsert_batches
from ingestion_sources import download_blob_to_file, resolve_blob_credentials
from utils.utils import read_yaml

log = logging.getLogger(__name__)

"""
    create_batches.py's batching pipeline (preprocess -> group into 3-hourly
    sub-batches -> assign batch folders), sourced from images/samples/locations
    in the DB instead of the CSV + find_most_recent_csv.

    Writes each batch's raws to cfg.paths.batches_root (downloaded from the
    weedsimagerepo blob, not azcopy'd blob-to-blob like the old
    field-batches-container version) - this *is* the "copy raws to NFS" step
    from the data flow, with batches_root pointed at a local test directory
    until it's pointed at the real NFS mount. Not part of the automatic
    pipeline (cfg.pipeline) yet - deliberately kept commented out there, since
    it writes real files. Run manually:
        python main.py general.task=create_batches_db +pipeline=[create_batches_db]

    Every run persists the computed plan into planned_batches (DB-only, no
    Azure/filesystem I/O) before attempting any downloads - query that table
    to see what's about to be created without needing to actually run the
    download step.

    cfg.create_batches.max_batches caps how many distinct batches actually get
    downloaded per run (oldest batch_date first) - planned_batches still shows
    the full backlog regardless. Override per run:
        python main.py general.task=create_batches_db +pipeline=[create_batches_db] create_batches.max_batches=10
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


def already_archived_base_names(conn) -> set:
    """base_names that already have a raw on NFS or JUNO (file_status, kept
    current by scan_file_locations), so filter_batched_data() knows what not
    to (re-)download - regardless of which of those two locations it's at.
    Requires file_status to be reasonably fresh: run scan_file_locations
    beforehand if raws may have landed on NFS/JUNO since the last scan."""
    rows = conn.execute(
        "SELECT base_name FROM file_status WHERE raw_in_nfs = 1 OR raw_in_juno = 1"
    ).fetchall()
    return {row[0] for row in rows}


def persist_planned_batches(conn, df: pd.DataFrame) -> int:
    """Materializes the current download plan (df after filter_batched_data -
    grouped into batches, not archived anywhere yet) into planned_batches, as
    a full replace - a snapshot of "what's about to be created", not an
    incremental log. Safe to call whether or not the download step actually
    runs afterward: no Azure/filesystem I/O, DB-only."""
    planned_at = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM planned_batches")
    conn.executemany(
        """
        INSERT INTO planned_batches
            (base_name, raw_blob_name, batch_label, location_code, batch_date, sub_batch_index, target_path, master_ref_id, planned_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row.BaseName,
                row.RawName,
                row.BatchID,
                row.UsState,
                row.CameraInfo_Date.strftime("%Y-%m-%d"),
                row.SubBatchDir,
                row.batches,
                row.MasterRefID,
                planned_at,
            )
            for row in df.itertuples(index=False)
        ],
    )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM planned_batches").fetchone()[0]
    log.info(f"Persisted {count} planned_batches rows")
    return count


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
        self.batches_root = cfg.paths.batches_root
        self.conn = get_connection(cfg.paths.db_path)
        self.read_and_convert_datetime()

    def config_keys(self) -> None:
        self.weedimgrepo_url, self.read_weedimgrepo_key, self.weedimgrepo_container = resolve_blob_credentials(
            self.ykeys, "weedsimagerepo"
        )
        log.debug("Configured weedsimagerepo read credentials.")

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

    def existing_raw_base_names_on_disk(self) -> set:
        """Ground-truth check against batches_root itself: for every distinct
        (UsState, CameraInfo_Date) batch among current candidates, lists every
        raw base_name that already exists anywhere under that batch's raws/
        tree (any sub-batch folder, not just the one a fresh run would
        naturally target). Unlike already_archived_base_names() (file_status,
        only as fresh as the last scan_file_locations run), this is always
        current as of right now - it's what actually prevents the same raw
        from being downloaded into two different sub-batch directories across
        two create_batches_db runs with no scan in between. One directory
        listing per candidate batch, not a full NFS crawl."""
        base_names = set()
        for us_state, date in self.df[["UsState", "CameraInfo_Date"]].drop_duplicates().itertuples(index=False):
            raws_dir = Path(self.batches_root) / f"{us_state}_{date.strftime('%Y-%m-%d')}" / "raws"
            if not raws_dir.is_dir():
                continue
            base_names.update(p.stem for p in raws_dir.glob("*/*") if p.suffix.lower() == ".arw")
        return base_names

    def preprocess_df(self) -> "DbBatchProcessor":
        log.info("Preprocessing DataFrame")
        self.df = self.df[self.df["HasMatchingJpgAndRaw"] == True].dropna(subset=["UsState"])
        self.df = self.df[self.df["Extension"] == "jpg"]
        self.df = self.df.dropna(subset="CameraInfo_DateTime")
        return self

    def resolve_available_sub_batch_dir(self, us_state: str, date, sub_batch_index_padded: str) -> str:
        """Finds a sub-batch directory name under batches_root that doesn't
        already have any files in it, so a run never plans to write into an
        already-populated directory - e.g. one a previous run already filled
        and someone has since preprocessed in RawTherapee, which a late
        straggler raw (its 3-hourly window recomputed fresh each run) would
        otherwise land back in. Tries the natural index, then <index>_1,
        <index>_2, ... Read-only (os.listdir/exists), never creates anything."""
        base = Path(self.batches_root) / f"{us_state}_{date.strftime('%Y-%m-%d')}" / "raws"
        candidate = sub_batch_index_padded
        suffix = 0
        while (base / candidate).exists() and any((base / candidate).iterdir()):
            suffix += 1
            candidate = f"{sub_batch_index_padded}_{suffix}"
        return candidate

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

        group_keys = self.df[["UsState", "CameraInfo_Date", "SubBatchIndex_Padded"]].drop_duplicates()
        resolved_dir_by_group = {
            (row.UsState, row.CameraInfo_Date, row.SubBatchIndex_Padded): self.resolve_available_sub_batch_dir(
                row.UsState, row.CameraInfo_Date, row.SubBatchIndex_Padded
            )
            for row in group_keys.itertuples(index=False)
        }
        self.df["SubBatchDir"] = self.df.apply(
            lambda row: resolved_dir_by_group[(row["UsState"], row["CameraInfo_Date"], row["SubBatchIndex_Padded"])],
            axis=1,
        )
        self.df["batches"] = self.df.apply(lambda row: f"{row['UsState']}_{row['CameraInfo_Date'].strftime('%Y-%m-%d')}/raws/{row['SubBatchDir']}/{row['RawName']}", axis=1)
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
        downloaded"), so images.batch_id reflects batch membership regardless of
        whether the download to batches_root has happened yet."""
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

    def filter_batched_data(self, archived_base_names: set) -> None:
        """Filters out base_names that already have a raw on NFS or JUNO, and
        raises on duplicate image names."""
        self.df = self.df[~self.df["BaseName"].isin(archived_base_names)]
        if len(self.df) == 0:
            log.info("No new images present. Nothing to download to batches_root. Exiting.")
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

    def limit_to_n_batches(self, max_batches) -> None:
        """Narrows self.df to at most max_batches distinct batches (oldest
        batch_date first), so a single run downloads a manageable slice of the
        backlog instead of all of it. Only affects what gets downloaded this
        run - planned_batches (persisted before this is called) still reflects
        the full candidate backlog regardless of this limit. max_batches=None
        means no limit."""
        if max_batches is None:
            return

        batch_dates = self.df[["BatchID", "CameraInfo_Date"]].drop_duplicates("BatchID").sort_values("CameraInfo_Date")
        selected = set(batch_dates["BatchID"].head(max_batches))
        self.df = self.df[self.df["BatchID"].isin(selected)]
        log.info(
            f"Limited this run to {len(selected)}/{len(batch_dates)} candidate batches "
            f"({len(self.df)} raws): {sorted(selected)}"
        )

    def download_raw_to_batches_root(self, batch: str) -> None:
        """Downloads one raw from weedsimagerepo and writes it to
        <batches_root>/<batch> (batch is e.g. 'TX_2024-07-07/raws/01/NAME.ARW'),
        creating parent directories as needed. Only ever writes under
        batches_root - never touches the source blob or any other location."""
        blob_name = Path(batch).name
        dest_path = Path(self.batches_root) / batch
        log.info(f"Downloading {blob_name} to {dest_path}")
        try:
            download_blob_to_file(
                self.weedimgrepo_url, self.read_weedimgrepo_key, self.weedimgrepo_container, blob_name, dest_path
            )
        except Exception:
            log.exception(f"Failed to download {blob_name} to {dest_path}")

    def process_df(self) -> None:
        log.info("Processing DataFrame without concurrency")
        for _, row in self.df.reset_index().iterrows():
            self.download_raw_to_batches_root(row["batches"])

    def process_df_concurrently(self) -> None:
        log.info("Processing DataFrame with concurrency")
        batches = self.df["batches"].unique()
        cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        max_workers = max(1, int(cpu_count / 3))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.download_raw_to_batches_root, batch) for batch in batches]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.error(f"Error processing batch: {e}")


def main(cfg: DictConfig) -> None:
    """Mirrors create_batches.py's main(), sourcing the batch DataFrame from the
    DB instead of the CSV, and writing raws to cfg.paths.batches_root instead
    of azcopy'ing to the field-batches container."""
    log.info(f"Starting {cfg.general.task}")
    log.info(f"Writing batches to {cfg.paths.batches_root}")

    dataproc = DbBatchProcessor(cfg)
    try:
        dataproc.config_keys()
        dataproc.split_datetime()
        dataproc.preprocess_df()

        # file_status can be stale between runs if scan_file_locations hasn't
        # been re-run - existing_raw_base_names_on_disk() is the always-fresh
        # ground-truth check that actually prevents the same raw being
        # downloaded into two different sub-batch directories across runs.
        archived_base_names = already_archived_base_names(dataproc.conn) | dataproc.existing_raw_base_names_on_disk()

        dataproc.adjust_groups()
        dataproc.warn_on_unknown_batch_labels()
        dataproc.persist_batches()
        dataproc.filter_batched_data(archived_base_names)
        persist_planned_batches(dataproc.conn, dataproc.df)
        dataproc.limit_to_n_batches(cfg.create_batches.max_batches)

        run_concurrent = True
        if run_concurrent:
            dataproc.process_df_concurrently()
        else:
            dataproc.process_df()
    finally:
        dataproc.conn.close()

    log.info(f"Task '{cfg.general.task}' completed successfully.")
