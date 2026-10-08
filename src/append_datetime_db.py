#!/usr/bin/env python3
# fmt: off
# isort: off
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from omegaconf import DictConfig

from db.connection import get_connection
from db.normalize import normalize_datetime
from db.upsert import update_image_exif_datetime
from utils.utils import download_azcopy, get_exif_data, read_yaml

log = logging.getLogger(__name__)

"""
    DB-driven equivalent of append_datetime.py: fills images.exif_datetime for
    images ingested since section 1's one-time historical migration, which is the
    only thing that has ever populated this column - nothing in the ongoing
    config-driven ingestion (sections 2-3) writes it, and append_datetime.py
    itself only ever wrote its results back to CSVs, never the DB.

    Two passes, cheapest first:
      1. Stem lookup: an image missing exif_datetime inherits it from a sibling
         image (same base_name) that already has one - no network needed.
      2. EXIF download: any JPG still missing it gets downloaded via azcopy and
         its EXIF DateTimeOriginal extracted (same mechanism append_datetime.py
         uses). Limited to JPGs, same as legacy - ARW EXIF isn't read directly.
      3. A second stem pass over the still-missing set, so an ARW whose sibling
         JPG was *also* missing before step 2 still gets filled once that JPG's
         datetime is known. append_datetime.py doesn't do this second pass - its
         single fill_missing_by_stem() call runs before the download step, so a
         pair that started out both-missing only gets the JPG side backfilled.

    Not part of the automatic pipeline (cfg.pipeline) yet - run manually:
        python main.py general.task=append_datetime_db +pipeline=[append_datetime_db]
    Optionally cap how many images get downloaded in one run (the stem passes
    are unlimited, since they're free) via:
        python main.py general.task=append_datetime_db +pipeline=[append_datetime_db] \
            +exif_download_limit=50
"""


def find_missing_exif(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT blob_name, base_name, extension, image_url FROM images WHERE exif_datetime IS NULL",
        conn,
    )


def known_jpg_exif_by_base_name(conn) -> dict:
    return dict(conn.execute(
        "SELECT base_name, exif_datetime FROM images WHERE extension = 'jpg' AND exif_datetime IS NOT NULL"
    ).fetchall())


def download_and_extract_exif(row, sas_token: str) -> tuple:
    try:
        img_url = row.image_url + sas_token
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            download_azcopy(img_url, tmp_file.name)
        exif = get_exif_data(tmp_file.name)
        os.remove(tmp_file.name)
        return row.blob_name, exif.get("EXIF DateTimeOriginal")
    except Exception as e:
        log.warning(f"EXIF extraction failed for {row.blob_name}: {e}")
        return row.blob_name, None


def download_fill(jpgs: pd.DataFrame, sas_token: str) -> dict:
    """Downloads and extracts EXIF for the given JPG rows. Returns
    {blob_name: raw EXIF datetime string or None}."""
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    max_workers = max(1, int(cpu_count / 3))
    log.info(f"Downloading {len(jpgs)} JPGs for EXIF recovery (max_workers={max_workers})...")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_and_extract_exif, row, sas_token) for row in jpgs.itertuples(index=False)]
        for future in as_completed(futures):
            blob_name, raw_exif_dt = future.result()
            results[blob_name] = raw_exif_dt
    return results


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    conn = get_connection(cfg.paths.db_path)
    try:
        missing = find_missing_exif(conn)
        base_name_of = dict(zip(missing["blob_name"], missing["base_name"]))
        log.info(f"{len(missing)} images missing exif_datetime")

        updates = {}

        # Pass 1: stem lookup against already-known jpg datetimes - free, no network.
        lookup = known_jpg_exif_by_base_name(conn)
        for row in missing.itertuples(index=False):
            if row.base_name in lookup:
                updates[row.blob_name] = lookup[row.base_name]
        log.info(f"Filled {len(updates)} via stem lookup (no network)")

        # Pass 2: download + extract EXIF for JPGs still missing.
        still_missing = missing[~missing["blob_name"].isin(updates)]
        jpgs = still_missing[(still_missing["extension"] == "jpg") & still_missing["image_url"].notna()]
        limit = cfg.get("exif_download_limit", None)
        if limit is not None:
            jpgs = jpgs.head(int(limit))
            log.info(f"exif_download_limit={limit}: downloading only the first {len(jpgs)} of "
                      f"{len(still_missing[(still_missing['extension'] == 'jpg') & still_missing['image_url'].notna()])} eligible JPGs")

        keys = read_yaml(cfg.pipeline_keys)
        sas_token = keys["blobs"]["weedsimagerepo"]["sas_token"]
        downloaded_raw = download_fill(jpgs, sas_token) if len(jpgs) else {}

        downloaded = {}
        for blob_name, raw in downloaded_raw.items():
            if not raw:
                continue
            normalized = normalize_datetime(pd.Series([raw])).iloc[0]
            if pd.notna(normalized):
                downloaded[blob_name] = normalized
        updates.update(downloaded)
        log.info(f"Filled {len(downloaded)} via EXIF download ({len(downloaded_raw) - len(downloaded)} downloads had no usable EXIF)")

        # Pass 3: second stem pass, so a sibling (e.g. ARW) of a just-downloaded
        # JPG that was ALSO missing before download still gets filled.
        newly_known_by_base = {base_name_of[bn]: dt for bn, dt in downloaded.items() if bn in base_name_of}
        still_missing2 = missing[~missing["blob_name"].isin(updates)]
        pass3_count = 0
        for row in still_missing2.itertuples(index=False):
            if row.base_name in newly_known_by_base:
                updates[row.blob_name] = newly_known_by_base[row.base_name]
                pass3_count += 1
        log.info(f"Filled {pass3_count} more via second stem pass (siblings of newly-downloaded JPGs)")

        if updates:
            update_image_exif_datetime(conn, updates)
            conn.commit()

        log.info(f"Total filled: {len(updates)}/{len(missing)}; still missing: {len(missing) - len(updates)}")
    finally:
        conn.close()
    log.info(f"{cfg.general.task} completed.")
