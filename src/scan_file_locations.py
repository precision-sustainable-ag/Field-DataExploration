import logging
from datetime import datetime, timezone
from pathlib import PurePosixPath

from omegaconf import DictConfig

from db.connection import get_connection
from db.locations import all_known_locations, batch_folder_regex
from db.reporting import refresh_file_status
from db.upsert import upsert_file_locations
from ingestion_sources import GlobusEndpointSource, NfsFilesystemSource

log = logging.getLogger(__name__)

"""
    Scans locations that aren't Azure (NFS long-term storage, SCINet JUNO via
    Globus) for raw/processed image files, driven by `cfg.sources` (entries
    with `type: filesystem` or `type: globus`), and upserts what it finds into
    `file_locations`. Blob presence is already captured by `images`
    (container + extension) so isn't duplicated here.

    Read-only against every source: lists/stats files, never writes, moves, or
    deletes anything at any of these locations.

    Not part of the automatic pipeline yet - run manually to validate first:
        python main.py general.task=scan_file_locations +pipeline=[scan_file_locations]
"""

ARTIFACT_KIND_BY_FOLDER = {
    "raws": "raw",
    "developed-images": "processed_jpg",
}


def classify_relative_path(relative_path: str) -> dict | None:
    """Maps a path like 'TX_2024-07-07/raws/01/NCA03585.ARW' or
    'TX_2024-07-07/developed-images/NCA03585.jpg' to {batch_label, base_name,
    extension, artifact_kind}, or None if it doesn't match the expected
    <batch>/raws|developed-images/... layout (e.g. NFS's stray '01'/'02'
    staging folders that sit alongside real batch folders but aren't one)."""
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

    return {
        "batch_label": batch_label,
        "base_name": base_name,
        "extension": extension,
        "artifact_kind": artifact_kind,
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
        all_rows = []
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
            all_rows.extend(rows)

        if all_rows:
            warn_on_unknown_batch_labels(conn, all_rows)
            upserted = upsert_file_locations(conn, all_rows)
            conn.commit()
            log.info(f"Upserted {upserted} file_locations rows")
        else:
            log.warning("No filesystem/globus sources produced rows, nothing to upsert")

        refresh_file_status(conn)
        conn.commit()
    finally:
        conn.close()

    log.info(f"{cfg.general.task} completed.")
