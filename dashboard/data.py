"""Load and classify the Field-AgIR image inventory from field_exploration.db.
Used by snapshot.py, which runs where the database is mounted (SUNNY).

Read-only: the database is opened with mode=ro and nothing is ever written to it.
"""
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = "/mnt/research-projects/r/raatwell/longterm_images3/field-db/field_exploration.db"

# Every image lands in exactly one state, checked in this order (first match wins).
# Each state belongs to one category.
STATE_CATEGORY = {
    "Cutout": "Have",
    "Annotated, no plant found": "Have",
    "No raw (never uploaded)": "Don't have",
    "No metadata": "Don't have",
    "Developed, not annotated": "To be processed",
    "Raw on NFS, not developed": "To be processed",
    "Raw in blob only": "To be processed",
}
CATEGORIES = ["Have", "To be processed", "Don't have"]

# Cutout statuses the classification recognises. Anything else on an image's
# first cutout row falls through to the raw/developed states and is listed on
# the report's Data issues tab so it can be reviewed.
PLANT_STATUSES = ["detected_segmented", "segmented"]
NO_PLANT_STATUSES = ["no_detection"]
KNOWN_STATUSES = PLANT_STATUSES + NO_PLANT_STATUSES

# Species recorded under two names that share a class_id (see cutout_table_schema.md).
SPECIES_ALIASES = {
    "common waterhemp": "waterhemp",
    "junglerice": "jungle rice",
    "crabgrass (large or other spp.)": "large crabgrass",
}

COVER_CROP_TYPE = "COVERCROPS"

IMAGES_QUERY = """
    SELECT fs.base_name, fs.master_ref_id, fs.location_code, fs.plant_type, fs.species,
           fs.batch_label, fs.raw_in_blob, fs.raw_in_nfs, fs.processed_jpg_in_nfs,
           s.master_ref_id IS NOT NULL AS has_metadata,
           c.status AS cutout_status, c.cutout_path
    FROM file_status fs
    LEFT JOIN samples s ON s.master_ref_id = fs.master_ref_id
    LEFT JOIN cutouts c ON c.base_name = fs.base_name AND c.cutout_index = 0
"""
# Joining images in SQL takes ~40 s over NFS; merging in pandas takes under a second.
CAPTURE_QUERY = """
    SELECT base_name, exif_datetime AS captured_at
    FROM images
    WHERE base_name IS NOT NULL AND exif_datetime IS NOT NULL
"""
CUTOUT_COUNT_QUERY = f"""
    SELECT base_name, COUNT(*) AS n_cutouts
    FROM cutouts
    WHERE status IN ({", ".join(f"'{s}'" for s in PLANT_STATUSES)})
    GROUP BY base_name
"""
SAMPLES_WITHOUT_IMAGES_QUERY = """
    SELECT master_ref_id, location_code, plant_type, species, {recorded} AS recorded_at
    FROM samples
    WHERE master_ref_id NOT IN (SELECT master_ref_id FROM file_status WHERE master_ref_id IS NOT NULL)
"""
# When a sample was entered in the field app. The master record's timestamp first,
# then the plant-type tables', for samples that only have one of those.
SAMPLE_TIMESTAMP_COLUMNS = [
    "wirmastermeta_timestamp", "wirweedsmeta_timestamp", "wircropsmeta_timestamp",
    "wircovercropsmeta_timestamp", "wirsoilsmeta_timestamp",
]
FILE_STATUS_COUNT_QUERY = "SELECT COUNT(*), COUNT(DISTINCT base_name), MAX(refreshed_at) FROM file_status"


def default_db_path() -> str:
    """The database on long-term storage, unless FIELD_DB points somewhere else."""
    return os.environ.get("FIELD_DB", DEFAULT_DB)


def classify(df: pd.DataFrame) -> np.ndarray:
    has_raw = df.raw_in_blob.eq(1) | df.raw_in_nfs.eq(1) | df.processed_jpg_in_nfs.eq(1)
    conditions = [
        df.cutout_status.isin(PLANT_STATUSES),
        df.cutout_status.isin(NO_PLANT_STATUSES),
        ~has_raw,
        df.has_metadata.eq(0),
        df.processed_jpg_in_nfs.eq(1),
        df.raw_in_nfs.eq(1),
    ]
    choices = list(STATE_CATEGORY)[:6]
    return np.select(conditions, choices, default="Raw in blob only")


def season(captured_at: pd.Series, plant_type: pd.Series) -> pd.Series:
    """Same rule as plot_by_season_db.py in Field-DataExploration: cover crops
    run October-September (labeled 2025/2026), everything else by calendar year."""
    year = captured_at.dt.year.astype("Int64")
    cover = plant_type.eq(COVER_CROP_TYPE)
    start = year.where(~(cover & captured_at.dt.month.lt(10)), year - 1)
    label = year.astype("string").mask(cover, start.astype("string") + "/" + (start + 1).astype("string"))
    return label.fillna("Unknown")


def clean_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["location_code", "plant_type", "species"]:
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA).fillna("Unknown")
    df["species"] = df["species"].replace(SPECIES_ALIASES)
    return df


def with_season(samples: pd.DataFrame) -> pd.DataFrame:
    """Season for samples, from when they were recorded (they have no capture time)."""
    recorded = pd.to_datetime(samples["recorded_at"], errors="coerce", utc=True).dt.tz_localize(None)
    samples["season"] = season(recorded, samples["plant_type"])
    return samples


def load(db_path: str) -> dict:
    """Return the per-image frame plus everything the report needs alongside it."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        images = pd.read_sql_query(IMAGES_QUERY, conn)
        captured = pd.read_sql_query(CAPTURE_QUERY, conn)
        cutout_counts = pd.read_sql_query(CUTOUT_COUNT_QUERY, conn)
        sample_columns = {row[1] for row in conn.execute("PRAGMA table_info(samples)")}
        stamps = [c for c in SAMPLE_TIMESTAMP_COLUMNS if c in sample_columns]
        recorded = f"COALESCE({', '.join(stamps)})" if len(stamps) > 1 else (stamps[0] if stamps else "NULL")
        samples_without_images = pd.read_sql_query(SAMPLES_WITHOUT_IMAGES_QUERY.format(recorded=recorded), conn)
        fs_rows, fs_distinct, refreshed_at = conn.execute(FILE_STATUS_COUNT_QUERY).fetchone()

    for column in ["raw_in_blob", "raw_in_nfs", "processed_jpg_in_nfs", "has_metadata"]:
        images[column] = images[column].fillna(0).astype(int)

    # A base_name can have a raw and a preview row in images; take the earliest capture time.
    captured["captured_at"] = pd.to_datetime(captured["captured_at"], errors="coerce")
    captured = captured.groupby("base_name", as_index=False)["captured_at"].min()
    df = images.merge(captured, on="base_name", how="left")
    df = df.merge(cutout_counts, on="base_name", how="left")
    df["n_cutouts"] = df["n_cutouts"].fillna(0).astype(int)

    df = clean_dimensions(df)
    df["state"] = classify(df)
    df["category"] = df["state"].map(STATE_CATEGORY)
    df["season"] = season(df["captured_at"], df["plant_type"])
    # Only cutout images contribute cutouts, so a stray row on another state can't inflate counts.
    df.loc[df.state != "Cutout", "n_cutouts"] = 0
    status = df["cutout_status"].astype("string")
    df["unrecognised_status"] = status.where(status.notna() & ~status.isin(KNOWN_STATUSES), "")

    checks = [
        {
            "check": "One row per file_status row",
            "what": "The report starts from file_status (one row per image) and attaches each image's sample "
                    "record and first cutout row. Afterwards there should still be exactly one row per image.",
            "expected": int(fs_rows),
            "actual": len(df),
            "ok": len(df) == fs_rows,
            "hint": "More rows than file_status means an image matched two sample records (duplicate "
                    "samples.master_ref_id) or two first cutout rows, so it was counted twice.",
        },
        {
            "check": "base_name is unique in file_status",
            "what": "base_name is the image's file name, its ID everywhere else in the database. "
                    "Each name should appear once in file_status.",
            "expected": int(fs_rows),
            "actual": int(fs_distinct),
            "ok": fs_rows == fs_distinct,
            "hint": "Fewer distinct names than rows means some images are listed twice and counted twice.",
        },
    ]
    return {
        "images": df,
        "samples_without_images": with_season(clean_dimensions(samples_without_images)),
        "refreshed_at": refreshed_at,
        "checks": checks,
    }
