import logging
from datetime import datetime, timedelta, timezone
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
#
# BatchLabel prefers other_presence.raw_nfs_batch_label (this run's live scan
# of the NFS folder name) over batches.batch_label (the label assigned once
# when the batch was created) wherever both exist: batches.batch_label is
# effectively write-once (db.upsert.upsert_batches inserts ON CONFLICT DO
# NOTHING) and isn't touched by db.locations.apply_batch_location_corrections
# even when it fixes that batch's location_code - only a real on-disk folder
# rename changes what scan_file_locations reports next run. Without this, a
# corrected-but-never-renamed batch (or a batch whose folder *was* renamed on
# disk) would keep showing its old, wrong label forever.
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
            MAX(CASE WHEN storage_location = 'nfs' AND artifact_kind = 'raw' THEN path END) AS raw_nfs_path,
            MAX(CASE WHEN storage_location = 'nfs' AND artifact_kind = 'raw' THEN sub_batch_index END) AS raw_nfs_sub_batch_index,
            MAX(CASE WHEN storage_location = 'nfs' AND artifact_kind = 'raw' THEN batch_label END) AS raw_nfs_batch_label
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
        samples.height AS Height,
        samples.size_class AS SizeClass,
        samples.growth_stage AS GrowthStage,
        samples.cotton_variety AS CottonVariety,
        samples.crop_or_fallow AS CropOrFallow,
        samples.crop_type_secondary AS CropTypeSecondary,
        samples.cover_crop_family AS CoverCropFamily,
        samples.flower_fruit_or_seeds AS FlowerFruitOrSeeds,
        samples.cloud_cover AS CloudCover,
        samples.ground_residue AS GroundResidue,
        samples.ground_cover AS GroundCover,
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
        CASE WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1
             THEN COALESCE(other_presence.raw_nfs_batch_label, batches.batch_label) END AS BatchLabel,
        CASE WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1 THEN other_presence.raw_nfs_sub_batch_index END AS SubBatchIndex,
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
        height, size_class, growth_stage, cotton_variety, crop_or_fallow,
        crop_type_secondary, cover_crop_family, flower_fruit_or_seeds,
        cloud_cover, ground_residue, ground_cover, username,
        has_matching_jpg_and_raw, raw_in_blob, preview_jpg_in_blob,
        raw_in_nfs, processed_jpg_in_nfs, raw_in_juno, processed_jpg_in_juno,
        needs_processing, ready_for_juno_upload, batch_id, batch_label, sub_batch_index, file_path, refreshed_at
    )
    SELECT
        blob_presence.base_name,
        blob_presence.master_ref_id,
        samples.location_code,
        samples.plant_type,
        samples.species,
        samples.height,
        samples.size_class,
        samples.growth_stage,
        samples.cotton_variety,
        samples.crop_or_fallow,
        samples.crop_type_secondary,
        samples.cover_crop_family,
        samples.flower_fruit_or_seeds,
        samples.cloud_cover,
        samples.ground_residue,
        samples.ground_cover,
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
        CASE WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1
             THEN COALESCE(other_presence.raw_nfs_batch_label, batches.batch_label) END,
        CASE WHEN COALESCE(other_presence.raw_in_nfs, 0) = 1 THEN other_presence.raw_nfs_sub_batch_index END,
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


# The join/column definitions live in the images_needing_processing SQL VIEW
# (schema.sql) so any tool querying the DB directly sees the same columns as
# this dataframe - just adds the ordering, which SQLite doesn't guarantee a
# plain `SELECT * FROM view` preserves on its own.
NEEDS_PROCESSING_QUERY = """
    SELECT * FROM images_needing_processing
    ORDER BY BatchLabel, SubBatchIndex, BaseName
"""


def load_needs_processing_dataframe(conn) -> pd.DataFrame:
    """One row per image with a raw on NFS but no processed_jpg counterpart
    yet, with batch_id/batch_label/sub_batch_index attached - run
    scan_file_locations (which calls refresh_file_status) first so this
    reflects the current NFS state. Same data as querying the
    images_needing_processing view directly, ordered."""
    return pd.read_sql_query(NEEDS_PROCESSING_QUERY, conn)


def export_needs_processing_csv(conn, csv_path: str) -> Path:
    """Writes load_needs_processing_dataframe()'s output to csv_path, same
    export-a-permanent-CSV convention as export_processing_status_csv."""
    df = load_needs_processing_dataframe(conn)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info(f"Exported {len(df)} rows to {csv_path}")
    return csv_path


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


# Sub-batch-level counterpart to PARTIAL_BATCH_SUMMARY_QUERY: same "some
# processed, not all" gap, broken out by the raws/<sub_batch_index>/ folder
# within a batch instead of the whole batch_label. Sourced from file_status
# (not file_locations, which doesn't track sub_batch_index) - same
# "known samples only" limitation as file_status generally (see its
# schema.sql comment).
PARTIAL_SUBBATCH_SUMMARY_QUERY = """
    WITH subbatch_counts AS (
        SELECT
            batch_id, batch_label, sub_batch_index,
            COUNT(*) AS raw_count,
            SUM(CASE WHEN processed_jpg_in_nfs = 1 THEN 1 ELSE 0 END) AS processed_count
        FROM file_status
        WHERE raw_in_nfs = 1
        GROUP BY batch_id, batch_label, sub_batch_index
    )
    SELECT
        batch_id AS BatchId,
        batch_label AS BatchLabel,
        sub_batch_index AS SubBatchIndex,
        raw_count AS RawCount,
        processed_count AS ProcessedCount,
        raw_count - processed_count AS MissingCount
    FROM subbatch_counts
    WHERE processed_count > 0 AND processed_count < raw_count
    ORDER BY MissingCount DESC
"""


def load_partial_subbatch_summary_dataframe(conn) -> pd.DataFrame:
    """One row per NFS sub-batch that's partially developed - some raws have
    a processed_jpg counterpart, not all - sorted worst-gap-first.
    Sub-batch-level counterpart to load_partial_batch_summary_dataframe."""
    return pd.read_sql_query(PARTIAL_SUBBATCH_SUMMARY_QUERY, conn)


# Sub-batch-level counterpart to NOT_STARTED_BATCH_SUMMARY_QUERY: raws
# present but zero processed JPGs, broken out by sub_batch_index.
NOT_STARTED_SUBBATCH_SUMMARY_QUERY = """
    SELECT
        batch_id AS BatchId,
        batch_label AS BatchLabel,
        sub_batch_index AS SubBatchIndex,
        COUNT(*) AS RawCount
    FROM file_status
    WHERE raw_in_nfs = 1
    GROUP BY batch_id, batch_label, sub_batch_index
    HAVING SUM(CASE WHEN processed_jpg_in_nfs = 1 THEN 1 ELSE 0 END) = 0
    ORDER BY RawCount DESC, BatchLabel, SubBatchIndex
"""


def load_not_started_subbatch_summary_dataframe(conn) -> pd.DataFrame:
    """One row per NFS sub-batch with raws present but zero processed JPGs -
    hasn't been touched in RawTherapee yet. Sub-batch-level counterpart to
    load_not_started_batch_summary_dataframe."""
    return pd.read_sql_query(NOT_STARTED_SUBBATCH_SUMMARY_QUERY, conn)


def export_not_started_subbatches_csv(conn, csv_path: str) -> Path:
    """Writes load_not_started_subbatch_summary_dataframe()'s output to
    csv_path, same export-a-permanent-CSV convention as
    export_needs_processing_csv - the full, untruncated backlog, since the
    Slack table version only shows what fits under MAX_TABLE_CHARS."""
    df = load_not_started_subbatch_summary_dataframe(conn)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info(f"Exported {len(df)} rows to {csv_path}")
    return csv_path


# Raws create_batches_db has grouped into a batch and assigned a target NFS
# path, but that don't have anything on NFS yet (not upserted into
# file_locations by a scan_file_locations run) - the "still needs to be
# created" backlog. Rolled up by state rather than one row per batch_label -
# the backlog can be hundreds of batches, too many for a Slack table (the
# per-batch detail is in planned_batches itself for anyone who needs it).
PLANNED_BATCHES_SUMMARY_QUERY = """
    SELECT
        location_code AS UsState,
        COUNT(DISTINCT batch_label) AS PlannedBatchCount,
        COUNT(*) AS PlannedRawCount
    FROM planned_batches
    GROUP BY location_code
    ORDER BY PlannedRawCount DESC
"""


def load_planned_batches_summary_dataframe(conn) -> pd.DataFrame:
    """One row per state with planned-but-not-yet-created NFS batches -
    raws already grouped/targeted by create_batches_db but not made yet."""
    return pd.read_sql_query(PLANNED_BATCHES_SUMMARY_QUERY, conn)


# Processed-image totals by species, sourced from the materialized
# file_status table (refreshed by scan_file_locations) rather than
# recomputing the presence join here.
PROCESSED_BY_SPECIES_QUERY = """
    SELECT
        COALESCE(species, 'Unknown') AS Species,
        COUNT(*) AS ProcessedCount
    FROM file_status
    WHERE processed_jpg_in_nfs = 1
    GROUP BY COALESCE(species, 'Unknown')
    ORDER BY ProcessedCount DESC
"""


def load_processed_by_species_dataframe(conn) -> pd.DataFrame:
    """One row per species with a count of processed_jpg_in_nfs=1 images -
    sum(ProcessedCount) is the overall total-processed-images figure."""
    return pd.read_sql_query(PROCESSED_BY_SPECIES_QUERY, conn)


# Processed-jpg files on NFS first discovered by a scan in the last
# since_days days - file_locations.first_seen_at is set once, on that row's
# first INSERT (db.upsert.upsert_file_locations), and never touched again by
# later scans, so it reflects when *we* first saw the file rather than the
# file's own on-disk mtime_utc. mtime_utc looked like the natural column here
# but isn't reliable for "newly processed": a copy onto NFS that preserves
# timestamps (rsync -a, cp -p) can carry an mtime from well before the file
# actually became visible to us, so a real "just processed" batch can have an
# mtime that already fell outside the window - or outside a future one - by
# the time a scan finds it, silently undercounting. first_seen_at can't be
# backdated like that: it's stamped at scan time, not copy time. Scoped to
# storage_location='nfs' since this report is specifically "newly processed
# in NFS" (see notify_slack.py) - JUNO rows share the same first_seen_at
# format now, so this is a deliberate report scope, not a format workaround.
NEWLY_PROCESSED_QUERY = """
    SELECT
        COALESCE(samples.location_code, 'Unknown') AS UsState,
        COUNT(*) AS ProcessedCount
    FROM file_locations
    LEFT JOIN samples ON file_locations.master_ref_id = samples.master_ref_id
    WHERE file_locations.storage_location = 'nfs'
      AND file_locations.artifact_kind = 'processed_jpg'
      AND file_locations.first_seen_at >= ?
    GROUP BY COALESCE(samples.location_code, 'Unknown')
    ORDER BY ProcessedCount DESC
"""


def load_newly_processed_dataframe(conn, since_days: int) -> pd.DataFrame:
    """Processed-jpg files on NFS first discovered by a scan in the last
    since_days days, by state - sum(ProcessedCount) is the "newly processed
    this week" headline figure for the weekly report."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    return pd.read_sql_query(NEWLY_PROCESSED_QUERY, conn, params=(cutoff,))


# Aggregated, not one-row-per-correction, since the weekly report wants "how
# often is the field app still getting UsState typed wrong and as what" -
# SUM(corrected_count) folds in any backfilled historical rows alongside the
# normal corrected_count=1 rows written per-entity going forward.
LOCATION_CODE_CORRECTIONS_QUERY = """
    SELECT
        entity_type AS EntityType,
        previous_code AS PreviousCode,
        corrected_code AS CorrectedCode,
        SUM(corrected_count) AS Count
    FROM location_code_corrections
    WHERE corrected_at >= ?
    GROUP BY entity_type, previous_code, corrected_code
    ORDER BY Count DESC
"""


def load_location_code_corrections_dataframe(conn, since_days: int) -> pd.DataFrame:
    """Location-code auto-corrections (db.locations.infer_location_codes_from_images
    / infer_batch_location_codes_from_images, applied by merge_samples.py's
    self-heal and fix_location_codes.py) in the last since_days days, grouped
    by (entity_type, previous_code -> corrected_code) - surfaces recurring
    metadata-entry errors (e.g. crews still typing 'DV' in the field app) in
    the weekly report instead of only being visible by querying
    location_code_corrections directly."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    return pd.read_sql_query(LOCATION_CODE_CORRECTIONS_QUERY, conn, params=(cutoff,))


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
