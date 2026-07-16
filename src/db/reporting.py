import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPORT_QUERY = """
    SELECT
        images.blob_name AS Name,
        images.base_name AS BaseName,
        images.extension AS Extension,
        images.size_mib AS SizeMiB,
        images.upload_datetime_utc AS UploadDateTimeUTC,
        images.exif_datetime AS CameraInfo_DateTime,
        images.image_url AS ImageURL,
        images.image_index AS ImageIndex,
        images.has_matching_jpg_and_raw AS HasMatchingJpgAndRaw,
        images.sub_batch_index AS SubBatchIndex,
        images.stem AS Stem,
        images.master_ref_id AS MasterRefID,
        samples.location_code AS UsState,
        samples.plant_type AS PlantType,
        samples.species AS Species,
        samples.username AS Username
    FROM images
    LEFT JOIN samples ON images.master_ref_id = samples.master_ref_id
"""


def load_report_dataframe(conn) -> pd.DataFrame:
    """The report/plotting column set (report.py, plot_by_season.py,
    image_inspection.py), sourced from images/samples instead of the CSV chain.
    Unlike the CSV, CameraInfo_DateTime is always present here - it's normalized
    at ingest (db/normalize.py), not bolted on later by append_datetime.py."""
    return pd.read_sql_query(REPORT_QUERY, conn)


# Shared by PROCESSING_STATUS_QUERY (CamelCase, for the pandas/CSV report) and
# _STATUS_TABLE_INSERT (snake_case, for the materialized file_status table) so
# the join/presence logic has one definition. One row per base_name (a
# raw+preview pair share a base_name, so this is base_name-level, not
# blob_name-level like REPORT_QUERY). Base names that only exist in
# file_locations (e.g. legacy NFS files never ingested from blob) aren't
# included - this reports processing/archival status for known samples, not
# an orphan-file audit.
_STATUS_CTES = """
    WITH blob_presence AS (
        SELECT
            base_name,
            MAX(master_ref_id) AS master_ref_id,
            MAX(has_matching_jpg_and_raw) AS has_matching_jpg_and_raw,
            MAX(batch_id) AS batch_id,
            MAX(CASE WHEN extension = 'arw' THEN 1 ELSE 0 END) AS raw_in_blob,
            MAX(CASE WHEN extension = 'jpg' THEN 1 ELSE 0 END) AS preview_jpg_in_blob
        FROM images
        WHERE base_name IS NOT NULL AND base_name != ''
        GROUP BY base_name
    ),
    other_presence AS (
        SELECT
            base_name,
            MAX(CASE WHEN storage_location = 'nfs' AND artifact_kind = 'raw' THEN 1 ELSE 0 END) AS raw_in_nfs,
            MAX(CASE WHEN storage_location = 'nfs' AND artifact_kind = 'processed_jpg' THEN 1 ELSE 0 END) AS processed_jpg_in_nfs,
            MAX(CASE WHEN storage_location = 'juno' AND artifact_kind = 'raw' THEN 1 ELSE 0 END) AS raw_in_juno,
            MAX(CASE WHEN storage_location = 'juno' AND artifact_kind = 'processed_jpg' THEN 1 ELSE 0 END) AS processed_jpg_in_juno,
            MAX(CASE WHEN storage_location = 'nfs' AND artifact_kind = 'raw' THEN path END) AS raw_nfs_path
        FROM file_locations
        GROUP BY base_name
    )
"""

_STATUS_FROM_JOIN = """
    FROM blob_presence
    LEFT JOIN other_presence ON blob_presence.base_name = other_presence.base_name
    LEFT JOIN samples ON blob_presence.master_ref_id = samples.master_ref_id
    LEFT JOIN batches ON blob_presence.batch_id = batches.id
"""

PROCESSING_STATUS_QUERY = f"""
    {_STATUS_CTES}
    SELECT
        blob_presence.base_name AS BaseName,
        blob_presence.master_ref_id AS MasterRefID,
        samples.location_code AS UsState,
        samples.plant_type AS PlantType,
        samples.species AS Species,
        samples.crop_or_fallow AS CropOrFallow,
        samples.cover_crop_family AS CoverCropFamily,
        samples.growth_stage AS GrowthStage,
        samples.username AS Username,
        blob_presence.has_matching_jpg_and_raw AS HasMatchingJpgAndRaw,
        blob_presence.raw_in_blob AS RawInBlob,
        blob_presence.preview_jpg_in_blob AS PreviewJpgInBlob,
        COALESCE(other_presence.raw_in_nfs, 0) AS RawInNfs,
        COALESCE(other_presence.processed_jpg_in_nfs, 0) AS ProcessedJpgInNfs,
        COALESCE(other_presence.raw_in_juno, 0) AS RawInJuno,
        COALESCE(other_presence.processed_jpg_in_juno, 0) AS ProcessedJpgInJuno,
        CASE
            WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1
                 AND COALESCE(other_presence.processed_jpg_in_nfs, 0) = 0
            THEN 1 ELSE 0
        END AS NeedsProcessing,
        CASE
            WHEN COALESCE(other_presence.processed_jpg_in_nfs, 0) = 1
                 AND COALESCE(other_presence.processed_jpg_in_juno, 0) = 0
            THEN 1 ELSE 0
        END AS ReadyForJunoUpload,
        CASE WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1 THEN blob_presence.batch_id END AS BatchId,
        CASE WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1 THEN batches.batch_label END AS BatchLabel,
        other_presence.raw_nfs_path AS FilePath
    {_STATUS_FROM_JOIN}
"""


def load_processing_status_dataframe(conn) -> pd.DataFrame:
    """Per-base_name presence/status across all 4 locations (Azure Blob, NFS,
    JUNO) joined with sample metadata - the "what's already processed, and
    where does it still need to go" view."""
    return pd.read_sql_query(PROCESSING_STATUS_QUERY, conn)


def export_processing_status_csv(conn, csv_path: str) -> Path:
    """Writes load_processing_status_dataframe()'s output to csv_path, same
    export-a-permanent-CSV convention as merge_samples.export_validation_csv."""
    df = load_processing_status_dataframe(conn)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info(f"Exported {len(df)} rows to {csv_path}")
    return csv_path


_STATUS_TABLE_INSERT = f"""
    {_STATUS_CTES}
    INSERT INTO file_status (
        base_name, master_ref_id, location_code, plant_type, species,
        crop_or_fallow, cover_crop_family, growth_stage, username,
        has_matching_jpg_and_raw, raw_in_blob, preview_jpg_in_blob,
        raw_in_nfs, processed_jpg_in_nfs, raw_in_juno, processed_jpg_in_juno,
        needs_processing, ready_for_juno_upload, batch_id, batch_label, file_path, refreshed_at
    )
    SELECT
        blob_presence.base_name,
        blob_presence.master_ref_id,
        samples.location_code,
        samples.plant_type,
        samples.species,
        samples.crop_or_fallow,
        samples.cover_crop_family,
        samples.growth_stage,
        samples.username,
        blob_presence.has_matching_jpg_and_raw,
        blob_presence.raw_in_blob,
        blob_presence.preview_jpg_in_blob,
        COALESCE(other_presence.raw_in_nfs, 0),
        COALESCE(other_presence.processed_jpg_in_nfs, 0),
        COALESCE(other_presence.raw_in_juno, 0),
        COALESCE(other_presence.processed_jpg_in_juno, 0),
        CASE
            WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1
                 AND COALESCE(other_presence.processed_jpg_in_nfs, 0) = 0
            THEN 1 ELSE 0
        END,
        CASE
            WHEN COALESCE(other_presence.processed_jpg_in_nfs, 0) = 1
                 AND COALESCE(other_presence.processed_jpg_in_juno, 0) = 0
            THEN 1 ELSE 0
        END,
        CASE WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1 THEN blob_presence.batch_id END,
        CASE WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1 THEN batches.batch_label END,
        other_presence.raw_nfs_path,
        ?
    {_STATUS_FROM_JOIN}
"""


def refresh_file_status(conn) -> int:
    """Materializes the same presence/status join as
    load_processing_status_dataframe() into the file_status table, as a full
    replace (this is a derived summary, not upserted incrementally row by
    row). Call after scan_file_locations/merge_samples so file_status reflects
    the latest scan."""
    refreshed_at = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM file_status")
    conn.execute(_STATUS_TABLE_INSERT, (refreshed_at,))
    count = conn.execute("SELECT COUNT(*) FROM file_status").fetchone()[0]
    log.info(f"Refreshed file_status: {count} rows")
    return count


# A batch counts as "partially developed" only if developed-images/ has *some*
# content (processed_count > 0) but not full coverage of raws/ - a batch with
# 0 processed images just hasn't been started yet, which is a different,
# expected backlog state, not a gap in already-claimed-done work.
PARTIAL_BATCH_SUMMARY_QUERY = """
    WITH raw_counts AS (
        SELECT batch_label, COUNT(DISTINCT base_name) AS raw_count
        FROM file_locations WHERE storage_location = 'nfs' AND artifact_kind = 'raw'
        GROUP BY batch_label
    ),
    processed_counts AS (
        SELECT batch_label, COUNT(DISTINCT base_name) AS processed_count
        FROM file_locations WHERE storage_location = 'nfs' AND artifact_kind = 'processed_jpg'
        GROUP BY batch_label
    )
    SELECT
        r.batch_label AS BatchLabel,
        r.raw_count AS RawCount,
        p.processed_count AS ProcessedCount,
        r.raw_count - p.processed_count AS MissingCount
    FROM raw_counts r
    JOIN processed_counts p ON r.batch_label = p.batch_label
    WHERE p.processed_count > 0 AND p.processed_count < r.raw_count
    ORDER BY MissingCount DESC
"""


def load_partial_batch_summary_dataframe(conn) -> pd.DataFrame:
    """One row per NFS batch that's partially developed - developed-images/
    has some processed JPGs but not one per raw - sorted worst-gap-first."""
    return pd.read_sql_query(PARTIAL_BATCH_SUMMARY_QUERY, conn)


# Batches with raws on NFS but zero processed JPGs - not started at all, as
# opposed to PARTIAL_BATCH_SUMMARY_QUERY's "started but incomplete".
NOT_STARTED_BATCH_SUMMARY_QUERY = """
    SELECT
        batch_label AS BatchLabel,
        COUNT(DISTINCT base_name) AS RawCount
    FROM file_locations
    WHERE storage_location = 'nfs' AND artifact_kind = 'raw'
    AND batch_label NOT IN (
        SELECT DISTINCT batch_label FROM file_locations
        WHERE storage_location = 'nfs' AND artifact_kind = 'processed_jpg'
    )
    GROUP BY batch_label
    ORDER BY BatchLabel
"""


def load_not_started_batch_summary_dataframe(conn) -> pd.DataFrame:
    """One row per NFS batch with raws present but zero processed JPGs -
    hasn't been touched in RawTherapee yet, sorted by batch label."""
    return pd.read_sql_query(NOT_STARTED_BATCH_SUMMARY_QUERY, conn)


def missing_processed_jpgs_for_batch(conn, batch_label: str) -> list:
    """base_names with a raw on NFS but no processed_jpg counterpart in the
    same batch, for one batch_label - the detail behind one row of
    load_partial_batch_summary_dataframe()."""
    rows = conn.execute(
        """
        SELECT r.base_name
        FROM file_locations r
        WHERE r.batch_label = ? AND r.storage_location = 'nfs' AND r.artifact_kind = 'raw'
        AND NOT EXISTS (
            SELECT 1 FROM file_locations p
            WHERE p.batch_label = r.batch_label AND p.storage_location = 'nfs'
            AND p.artifact_kind = 'processed_jpg' AND p.base_name = r.base_name
        )
        ORDER BY r.base_name
        """,
        (batch_label,),
    ).fetchall()
    return [row[0] for row in rows]
