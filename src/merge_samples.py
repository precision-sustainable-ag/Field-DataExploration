#!/usr/bin/env python3
# fmt: off
# isort: off
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from omegaconf import DictConfig

from db.connection import get_connection
from db.upsert import upsert_locations

log = logging.getLogger(__name__)

"""
    Declarative merge: coalesces raw_sample_attributes (staged by
    wir_table_generator.py) into samples, per cfg.coalesce_fields, and computes
    HasMatchingJpgAndRaw on images. Not part of the automatic pipeline
    (cfg.pipeline) yet - run manually and validate against
    process_tables_analysis.py's merged_blobs_tables_metadata.csv:
        python main.py general.task=merge_samples +pipeline=[merge_samples]
"""

FIELD_TO_COLUMN = {
    "UsState": "location_code",
    "PlantType": "plant_type",
    "CloudCover": "cloud_cover",
    "GroundResidue": "ground_residue",
    "GroundCover": "ground_cover",
    "Username": "username",
    "CoverCropFamily": "cover_crop_family",
    "GrowthStage": "growth_stage",
    "CottonVariety": "cotton_variety",
    "CropOrFallow": "crop_or_fallow",
    "CropTypeSecondary": "crop_type_secondary",
    "Species": "species",
    "Height": "height",
    "SizeClass": "size_class",
    "FlowerFruitOrSeeds": "flower_fruit_or_seeds",
}

SIZE_CLASS_NORMALIZATION = {
    "Large": "LARGE", "Medium": "MEDIUM", "Small": "SMALL",
    "3": "LARGE", "2": "MEDIUM", "1": "SMALL",
}

LINEAGE_SOURCES = ["wirmastermeta", "wircovercropsmeta", "wircropsmeta", "wirweedsmeta", "wirsoilsmeta"]

VALIDATION_CSV_QUERY = """
    SELECT
        images.blob_name AS Name,
        images.size_mib AS SizeMiB,
        images.upload_datetime_utc AS UploadDateTimeUTC,
        images.master_ref_id AS MasterRefID,
        images.image_url AS ImageURL,
        images.image_index AS ImageIndex,
        samples.location_code AS UsState,
        samples.plant_type AS PlantType,
        samples.cloud_cover AS CloudCover,
        samples.ground_residue AS GroundResidue,
        samples.ground_cover AS GroundCover,
        samples.username AS Username,
        samples.cover_crop_family AS CoverCropFamily,
        samples.growth_stage AS GrowthStage,
        samples.cotton_variety AS CottonVariety,
        samples.crop_or_fallow AS CropOrFallow,
        samples.crop_type_secondary AS CropTypeSecondary,
        samples.species AS Species,
        samples.height AS Height,
        samples.size_class AS SizeClass,
        samples.flower_fruit_or_seeds AS FlowerFruitOrSeeds,
        images.base_name AS BaseName,
        images.extension AS Extension,
        images.has_matching_jpg_and_raw AS HasMatchingJpgAndRaw,
        images.wirimagerefs_rowkey AS Wirimagerefs_rowkey,
        images.wirimagerefs_timestamp AS Wirimagerefs_timestamp,
        samples.wirmastermeta_rowkey AS Wirmastermeta_rowkey,
        samples.wirmastermeta_timestamp AS Wirmastermeta_timestamp,
        samples.wircovercropsmeta_rowkey AS Wircovercropsmeta_rowkey,
        samples.wircovercropsmeta_timestamp AS Wircovercropsmeta_timestamp,
        samples.wircropsmeta_rowkey AS Wircropsmeta_rowkey,
        samples.wircropsmeta_timestamp AS Wircropsmeta_timestamp,
        samples.wirweedsmeta_rowkey AS Wirweedsmeta_rowkey,
        samples.wirweedsmeta_timestamp AS Wirweedsmeta_timestamp
    FROM images
    LEFT JOIN samples ON images.master_ref_id = samples.master_ref_id
"""


def load_raw_sample_attributes(conn) -> dict:
    """master_ref_id -> {source: {"data": dict, "row_key": str, "timestamp": str}}"""
    grouped = defaultdict(dict)
    for source, master_ref_id, row_key, source_timestamp, data in conn.execute(
        "SELECT source, master_ref_id, row_key, source_timestamp, data FROM raw_sample_attributes"
    ):
        grouped[master_ref_id][source] = {
            "data": json.loads(data),
            "row_key": row_key,
            "timestamp": source_timestamp,
        }
    return grouped


def ensure_locations(conn, state_list, rollups=None) -> None:
    codes_in_data = set()
    for (data,) in conn.execute(
        "SELECT data FROM raw_sample_attributes WHERE source = 'wirmastermeta'"
    ):
        us_state = json.loads(data).get("UsState")
        if us_state:
            codes_in_data.add(us_state)
    upsert_locations(conn, state_list, codes_in_data, rollups)


def coalesce_sample(source_rows: dict, coalesce_fields) -> dict:
    result = {}
    for target_field, fallbacks in coalesce_fields.items():
        value = None
        for fallback in fallbacks:
            row = source_rows.get(fallback["source"])
            if row and row["data"].get(fallback["field"]):
                value = row["data"][fallback["field"]]
                break
        result[target_field] = value
    return result


def merge_sample(master_ref_id: str, source_rows: dict, coalesce_fields) -> tuple:
    coalesced = coalesce_sample(source_rows, coalesce_fields)

    size_class = coalesced.get("SizeClass")
    coalesced["SizeClass"] = SIZE_CLASS_NORMALIZATION.get(size_class, size_class)
    if coalesced.get("Species"):
        coalesced["Species"] = coalesced["Species"].lower()

    columns = ["master_ref_id"]
    values = [master_ref_id]
    for field, column in FIELD_TO_COLUMN.items():
        columns.append(column)
        values.append(coalesced.get(field))
    for source in LINEAGE_SOURCES:
        row = source_rows.get(source)
        columns.append(f"{source}_rowkey")
        columns.append(f"{source}_timestamp")
        values.append(row["row_key"] if row else None)
        values.append(row["timestamp"] if row else None)

    return columns, values


def upsert_samples(conn, source_rows_by_master_ref_id: dict, coalesce_fields) -> int:
    columns = None
    all_values = []
    for master_ref_id, source_rows in source_rows_by_master_ref_id.items():
        cols, values = merge_sample(master_ref_id, source_rows, coalesce_fields)
        columns = cols
        all_values.append(values)

    if not all_values:
        return 0

    placeholders = ", ".join(["?"] * len(columns))
    update_clause = ", ".join(f"{col}=excluded.{col}" for col in columns[1:])
    conn.executemany(
        f"""
        INSERT INTO samples ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT(master_ref_id) DO UPDATE SET {update_clause}
        """,
        all_values,
    )
    return len(all_values)


def update_has_matching_jpg_and_raw(conn) -> None:
    rows = conn.execute("SELECT id, base_name, extension FROM images").fetchall()
    extensions_by_base_name = defaultdict(set)
    for _, base_name, extension in rows:
        if extension:
            extensions_by_base_name[base_name].add(extension.lower())

    updates = [
        (int({"jpg", "arw"} <= extensions_by_base_name[base_name]), image_id)
        for image_id, base_name, _ in rows
    ]
    conn.executemany("UPDATE images SET has_matching_jpg_and_raw = ? WHERE id = ?", updates)
    log.info(f"Updated has_matching_jpg_and_raw for {len(updates)} images")


def export_validation_csv(conn, csv_path: str) -> None:
    cursor = conn.execute(VALIDATION_CSV_QUERY)
    columns = [description[0] for description in cursor.description]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(cursor.fetchall())
    log.info(f"Exported validation CSV to {csv_path}")


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    conn = get_connection(cfg.paths.db_path)
    try:
        ensure_locations(conn, cfg.state_list, cfg.get("location_rollups"))

        source_rows_by_master_ref_id = load_raw_sample_attributes(conn)
        num_samples = upsert_samples(conn, source_rows_by_master_ref_id, cfg.coalesce_fields)
        log.info(f"Coalesced {num_samples} samples")

        update_has_matching_jpg_and_raw(conn)

        conn.commit()

        csv_path = Path(cfg.paths.processed_datadir, "merged_blobs_tables_metadata_from_db.csv")
        export_validation_csv(conn, csv_path)
    finally:
        conn.close()
    log.info(f"{cfg.general.task} completed.")
