#!/usr/bin/env python3
# fmt: off
# isort: off
import logging

import pandas as pd
from omegaconf import DictConfig

from db.connection import get_connection
from db.normalize import normalize_datetime
from db.upsert import upsert_batches, upsert_locations
from utils.utils import read_csv_as_df

log = logging.getLogger(__name__)

"""
    One-time migration of the permanent merged CSV into the SQLite datastore.
    Not part of the automatic pipeline (cfg.pipeline) - run manually:
        python main.py general.task=migrate_to_db +pipeline=[migrate_to_db]
"""

SAMPLE_COLUMNS = {
    "UsState": "location_code",
    "PlantType": "plant_type",
    "Species": "species",
    "Height": "height",
    "SizeClass": "size_class",
    "GrowthStage": "growth_stage",
    "CottonVariety": "cotton_variety",
    "CropOrFallow": "crop_or_fallow",
    "CropTypeSecondary": "crop_type_secondary",
    "CoverCropFamily": "cover_crop_family",
    "FlowerFruitOrSeeds": "flower_fruit_or_seeds",
    "CloudCover": "cloud_cover",
    "GroundResidue": "ground_residue",
    "GroundCover": "ground_cover",
    "Username": "username",
    "Wirmastermeta_rowkey": "wirmastermeta_rowkey",
    "Wirmastermeta_timestamp": "wirmastermeta_timestamp",
    "Wircovercropsmeta_rowkey": "wircovercropsmeta_rowkey",
    "Wircovercropsmeta_timestamp": "wircovercropsmeta_timestamp",
    "Wircropsmeta_rowkey": "wircropsmeta_rowkey",
    "Wircropsmeta_timestamp": "wircropsmeta_timestamp",
    "Wirweedsmeta_rowkey": "wirweedsmeta_rowkey",
    "Wirweedsmeta_timestamp": "wirweedsmeta_timestamp",
}

IMAGE_COLUMNS = {
    "Name": "blob_name",
    "BaseName": "base_name",
    "Extension": "extension",
    "SizeMiB": "size_mib",
    "UploadDateTimeUTC": "upload_datetime_utc",
    "CameraInfo_DateTime": "exif_datetime",
    "ImageURL": "image_url",
    "ImageIndex": "image_index",
    "HasMatchingJpgAndRaw": "has_matching_jpg_and_raw",
    "SubBatchIndex": "sub_batch_index",
    "Stem": "stem",
    "Wirimagerefs_rowkey": "wirimagerefs_rowkey",
    "Wirimagerefs_timestamp": "wirimagerefs_timestamp",
}


def upsert_samples(conn, df: pd.DataFrame) -> int:
    df = df[df["MasterRefID"].notna()]
    sample_df = df[["MasterRefID", *SAMPLE_COLUMNS.keys()]].groupby(
        "MasterRefID", as_index=False
    ).first()
    sample_df = sample_df.rename(columns=SAMPLE_COLUMNS)

    rows = [
        (
            row.MasterRefID,
            *(getattr(row, col) if not pd.isna(getattr(row, col)) else None
              for col in SAMPLE_COLUMNS.values()),
        )
        for row in sample_df.itertuples(index=False)
    ]
    columns = ", ".join(["master_ref_id", *SAMPLE_COLUMNS.values()])
    placeholders = ", ".join(["?"] * (len(SAMPLE_COLUMNS) + 1))
    update_clause = ", ".join(
        f"{col}=excluded.{col}" for col in SAMPLE_COLUMNS.values()
    )
    conn.executemany(
        f"""
        INSERT INTO samples ({columns})
        VALUES ({placeholders})
        ON CONFLICT(master_ref_id) DO UPDATE SET {update_clause}
        """,
        rows,
    )
    log.info(f"Upserted {len(rows)} samples")
    return len(rows)


def upsert_images(conn, df: pd.DataFrame, batch_ids: dict) -> int:
    df = df[df["Name"].notna()]
    rows = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        values = [row_dict.get(csv_col) for csv_col in IMAGE_COLUMNS]
        values = [None if pd.isna(v) else v for v in values]
        master_ref_id = row_dict.get("MasterRefID")
        master_ref_id = None if pd.isna(master_ref_id) else master_ref_id
        batch_id = batch_ids.get(row_dict.get("BatchID"))
        rows.append((*values, master_ref_id, batch_id))

    ordered_db_cols = list(IMAGE_COLUMNS.values())
    columns = ", ".join([*ordered_db_cols, "master_ref_id", "batch_id"])
    placeholders = ", ".join(["?"] * (len(ordered_db_cols) + 2))
    update_clause = ", ".join(
        f"{col}=excluded.{col}" for col in [*ordered_db_cols[1:], "master_ref_id", "batch_id"]
    )
    conn.executemany(
        f"""
        INSERT INTO images ({columns})
        VALUES ({placeholders})
        ON CONFLICT(blob_name) DO UPDATE SET {update_clause}
        """,
        rows,
    )
    log.info(f"Upserted {len(rows)} images")
    return len(rows)


def main(cfg: DictConfig) -> None:
    csv_path = cfg.paths.permanent_merged_table
    log.info(f"Reading {csv_path}")
    df = read_csv_as_df(csv_path)
    total_rows = len(df)

    df["UploadDateTimeUTC"] = normalize_datetime(df["UploadDateTimeUTC"])
    df["CameraInfo_DateTime"] = normalize_datetime(df["CameraInfo_DateTime"])

    conn = get_connection(cfg.paths.db_path)
    try:
        upsert_locations(conn, cfg.state_list, set(df["UsState"].dropna().unique()))
        num_samples = upsert_samples(conn, df)
        batch_ids = upsert_batches(conn, df)
        num_images = upsert_images(conn, df, batch_ids)
        conn.commit()
    finally:
        conn.close()

    skipped = total_rows - num_images
    log.info(
        f"Migration complete: {total_rows} CSV rows, {num_images} images migrated, "
        f"{num_samples} distinct samples, {skipped} rows skipped (missing blob name)"
    )
