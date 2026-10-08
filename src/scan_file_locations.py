import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from omegaconf import DictConfig

from db.connection import get_connection
from db.locations import all_known_locations, batch_folder_regex
from db.reporting import export_needs_processing_csv, export_not_started_subbatches_csv, refresh_file_status
from db.upsert import delete_stale_file_locations, upsert_file_locations
from ingestion_sources import GlobusEndpointSource, NfsFilesystemSource

log = logging.getLogger(__name__)

"""
    Scans locations that aren't Azure (NFS long-term storage, SCINet JUNO via
    Globus) for raw/processed image files, driven by `cfg.sources` (entries
    with `type: filesystem` or `type: globus`), and reconciles what it finds
    into `file_locations`: upserts current files, then deletes any previously
    known row for that storage_location whose path wasn't seen this run (a
    file renamed or removed on disk since the last scan - see
    db.upsert.delete_stale_file_locations). Blob presence is already
    captured by `images` (container + extension) so isn't duplicated here.

    Read-only against every source itself: lists/stats files, never writes,
    moves, or deletes anything at any of these locations - only file_locations
    rows in our own DB are added/removed to match what's actually there.

    Runs weekly from scripts/run_weekly_report.sh. To run it on its own:
        python main.py general.task=scan_file_locations +pipeline=[scan_file_locations]
"""

ARTIFACT_KIND_BY_FOLDER = {
    "raws": "raw",
    "developed-images": "processed_jpg",
}


def classify_relative_path(relative_path: str) -> dict | None:
    """Maps a path like 'TX_2024-07-07/raws/01/NCA03585.ARW' or
    'TX_2024-07-07/developed-images/NCA03585.jpg' to {batch_label, base_name,
    extension, artifact_kind, sub_batch_index}, or None if it doesn't match
    the expected <batch>/raws|developed-images/... layout (e.g. NFS's stray
    '01'/'02' staging folders that sit alongside real batch folders but
    aren't one). sub_batch_index is the raws/<sub_batch_index>/ folder name
    (matching create_batches_db's SubBatchDir, e.g. '01' or '01_1' for a
    collision) - only set for artifact_kind='raw', since developed-images/
    is flat with no sub-batch folders."""
    parts = PurePosixPath(relative_path).parts
    if len(parts) < 3:
        return None

    batch_label, kind_folder, file_name = parts[0], parts[1], parts[-1]
    artifact_kind = ARTIFACT_KIND_BY_FOLDER.get(kind_folder)
    if artifact_kind is None:
        return None

    base_name, _, extension = file_name.rpartition(".")
    extension = extension.lower()
    if not base_name or extension not in ("arw", "jpg"):
        return None

    sub_batch_index = parts[2] if artifact_kind == "raw" and len(parts) >= 4 else None

    return {
        "batch_label": batch_label,
        "base_name": base_name,
        "extension": extension,
        "artifact_kind": artifact_kind,
        "sub_batch_index": sub_batch_index,
    }


def build_rows(entries: list, storage_location: str, scanned_at: str) -> list:
    rows = []
    for entry in entries:
        classified = classify_relative_path(entry["relative_path"])
        if classified is None:
            continue
        rows.append(
            {
                **classified,
                "storage_location": storage_location,
                "path": entry["path"],
                "size_bytes": entry.get("size_bytes"),
                "mtime_utc": entry.get("mtime_utc"),
                "scanned_at": scanned_at,
            }
        )
    return rows


def warn_on_unknown_batch_labels(conn, rows: list) -> None:
    """Flags batch labels that don't match any known location code, same
    pattern as create_batches_db.py's warn_on_unknown_batch_labels - doesn't
    drop the rows, just surfaces them since an unrecognized prefix usually
    means a new/renamed state code the DB doesn't know about yet."""
    locations = all_known_locations(conn)
    regex = batch_folder_regex(locations)
    unknown = sorted({row["batch_label"] for row in rows if not regex.match(row["batch_label"])})
    if unknown:
        log.warning(
            f"{len(unknown)} batch labels don't match any known location code: "
            f"{unknown[:10]}{'...' if len(unknown) > 10 else ''}"
        )


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    scanned_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection(cfg.paths.db_path)
    try:
        rows_by_storage_location = defaultdict(list)
        for source in cfg.sources:
            if source.type == "filesystem":
                entries = NfsFilesystemSource(source.name, source.root).fetch()
                storage_location = source.get("storage_location", source.name)
            elif source.type == "globus":
                if not source.get("root_path"):
                    log.warning(f"{source.name}: no root_path configured yet, skipping")
                    continue
                entries = GlobusEndpointSource(source.name, source.endpoint_id, source.root_path).fetch()
                storage_location = source.get("storage_location", source.name)
            else:
                continue

            if not entries:
                log.warning(f"{source.name} found no files, skipping")
                continue

            rows = build_rows(entries, storage_location, scanned_at)
            log.info(f"{source.name}: classified {len(rows)}/{len(entries)} files into file_locations rows")
            rows_by_storage_location[storage_location].extend(rows)

        all_rows = [row for rows in rows_by_storage_location.values() for row in rows]
        if all_rows:
            warn_on_unknown_batch_labels(conn, all_rows)
            upserted = upsert_file_locations(conn, all_rows)
            conn.commit()
            log.info(f"Upserted {upserted} file_locations rows")

            # Only for storage_locations that actually produced entries above -
            # a source skipped this run (empty/missing root) must never wipe
            # out a different, healthy source's rows.
            for storage_location, rows in rows_by_storage_location.items():
                current_paths = {row["path"] for row in rows}
                deleted = delete_stale_file_locations(conn, storage_location, current_paths)
                if deleted:
                    log.info(
                        f"Removed {deleted} stale file_locations rows for {storage_location} "
                        "(no longer found on disk this scan)"
                    )
            conn.commit()
        else:
            log.warning("No filesystem/globus sources produced rows, nothing to upsert")

        refresh_file_status(conn)
        conn.commit()

        # Written here, not report_db (which runs earlier in cfg.pipeline),
        # so this reflects this run's fresh scan rather than last week's.
        csv_path = Path(cfg.paths.reportdir_timestamp) / "images_needing_processing.csv"
        export_needs_processing_csv(conn, csv_path)

        not_started_csv_path = Path(cfg.paths.reportdir_timestamp) / "not_started_subbatches.csv"
        export_not_started_subbatches_csv(conn, not_started_csv_path)
    finally:
        conn.close()

    log.info(f"{cfg.general.task} completed.")
