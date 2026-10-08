import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _split_name(blob_name: str) -> tuple:
    base_name, _, extension = blob_name.rpartition(".")
    return (base_name or blob_name), extension.lower()


def build_locations(codes: set, rollups: dict = None) -> list:
    """Assigns parent_code only for codes explicitly listed in `rollups`
    (child_code -> parent_code, e.g. {"NC01": "NC"}), not by pattern-matching
    every '<letters><digits>' code. A blanket regex here previously also
    caught TX01/TX02 and rolled them into TX, even though those are genuinely
    separate locations (not a sub-code of TX the way NC01 is a sub-code of
    NC) - explicit beats inferred for something this consequential."""
    rollups = rollups or {}
    codes = sorted(codes)
    rows = []
    for code in codes:
        parent_code = rollups.get(code)
        if parent_code and parent_code not in codes:
            parent_code = None
        rows.append((code, code, parent_code))
    return rows


def upsert_locations(conn, state_list: list, codes_in_data: set, rollups: dict = None) -> None:
    missing_from_config = codes_in_data - set(state_list)
    if missing_from_config:
        log.warning(
            f"Location codes present in data but missing from cfg.state_list, "
            f"adding them to locations anyway: {sorted(missing_from_config)}"
        )

    rows = build_locations(set(state_list) | codes_in_data, rollups)
    conn.executemany(
        """
        INSERT INTO locations (code, display_name, parent_code)
        VALUES (?, ?, ?)
        ON CONFLICT(code) DO UPDATE SET
            display_name=excluded.display_name,
            parent_code=excluded.parent_code
        """,
        rows,
    )
    log.info(f"Upserted {len(rows)} locations")


def upsert_image_from_blob(conn, blob: dict) -> None:
    blob_name = blob["name"]
    base_name, extension = _split_name(blob_name)
    creation_time = blob["creation_time_utc"]
    upload_datetime_utc = creation_time.strftime("%Y-%m-%d %H:%M:%S") if creation_time else None

    conn.execute(
        """
        INSERT INTO images (blob_name, container, base_name, extension, size_mib, upload_datetime_utc)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(blob_name) DO UPDATE SET
            container=excluded.container,
            base_name=excluded.base_name,
            extension=excluded.extension,
            size_mib=excluded.size_mib,
            upload_datetime_utc=excluded.upload_datetime_utc
        """,
        (blob_name, blob["container"], base_name, extension, blob["memory_mb"], upload_datetime_utc),
    )


def upsert_image_from_imageref(conn, entity: dict) -> bool:
    master_ref_id = entity.get("MasterRefID")
    if not master_ref_id:
        return False

    blob_name = Path(entity["ImageURL"]).name
    conn.execute("INSERT OR IGNORE INTO samples (master_ref_id) VALUES (?)", (master_ref_id,))
    conn.execute(
        """
        INSERT INTO images (blob_name, master_ref_id, image_url, wirimagerefs_rowkey, wirimagerefs_timestamp)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(blob_name) DO UPDATE SET
            master_ref_id=excluded.master_ref_id,
            image_url=excluded.image_url,
            wirimagerefs_rowkey=excluded.wirimagerefs_rowkey,
            wirimagerefs_timestamp=excluded.wirimagerefs_timestamp
        """,
        (blob_name, master_ref_id, entity["ImageURL"], entity.get("RowKey"), entity.get("Timestamp")),
    )
    return True


def upsert_sample_attributes(conn, source_name: str, entity: dict, ingested_at: str, master_ref_key: str = "MasterRefID") -> bool:
    master_ref_id = entity.get(master_ref_key)
    if not master_ref_id:
        return False

    data = json.dumps({k: (v if v is None else str(v)) for k, v in entity.items()})
    conn.execute(
        """
        INSERT INTO raw_sample_attributes
            (source, master_ref_id, partition_key, row_key, source_timestamp, data, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, master_ref_id) DO UPDATE SET
            partition_key=excluded.partition_key,
            row_key=excluded.row_key,
            source_timestamp=excluded.source_timestamp,
            data=excluded.data,
            ingested_at=excluded.ingested_at
        """,
        (
            source_name,
            master_ref_id,
            entity.get("PartitionKey"),
            entity.get("RowKey"),
            entity.get("Timestamp"),
            data,
            ingested_at,
        ),
    )
    return True


def update_image_exif_datetime(conn, exif_datetime_by_blob_name: dict) -> int:
    """Sets images.exif_datetime for the given {blob_name: canonical datetime string}
    map. Only column owned here - doesn't touch any other image field."""
    conn.executemany(
        "UPDATE images SET exif_datetime = ? WHERE blob_name = ?",
        [(value, blob_name) for blob_name, value in exif_datetime_by_blob_name.items()],
    )
    return len(exif_datetime_by_blob_name)


def upsert_batches(conn, df: pd.DataFrame) -> dict:
    """Upserts one `batches` row per distinct df['BatchID'] label
    ('{location_code}_{date}'). Returns a mapping of batch_label -> batch id.
    Shared by migrate_to_db.py (historical BatchID column) and
    create_batches_db.py (freshly-computed batch assignments) so both upsert
    through one implementation instead of two."""
    known_codes = {row[0] for row in conn.execute("SELECT code FROM locations").fetchall()}
    labels = df["BatchID"].dropna().unique().tolist()
    rows = []
    unknown_location_labels = []
    for label in labels:
        location_code, _, batch_date = label.rpartition("_")
        if location_code not in known_codes:
            unknown_location_labels.append(label)
            location_code = None
        rows.append((location_code, label, batch_date or None))

    if unknown_location_labels:
        log.warning(
            f"{len(unknown_location_labels)} BatchID labels have a location prefix "
            f"that isn't a known location code, storing with location_code=NULL: "
            f"{unknown_location_labels[:10]}{'...' if len(unknown_location_labels) > 10 else ''}"
        )

    conn.executemany(
        """
        INSERT INTO batches (location_code, batch_label, batch_date)
        VALUES (?, ?, ?)
        ON CONFLICT(location_code, batch_date, batch_label) DO NOTHING
        """,
        rows,
    )
    log.info(f"Upserted {len(rows)} batches")

    label_to_id = {
        label: batch_id
        for batch_id, label in conn.execute(
            "SELECT id, batch_label FROM batches"
        ).fetchall()
    }
    return label_to_id


def update_image_batch_id(conn, batch_id_by_blob_name: dict) -> int:
    """Sets images.batch_id for the given {blob_name: batch id} map."""
    conn.executemany(
        "UPDATE images SET batch_id = ? WHERE blob_name = ?",
        [(batch_id, blob_name) for blob_name, batch_id in batch_id_by_blob_name.items()],
    )
    return len(batch_id_by_blob_name)


def upsert_file_locations(conn, rows: list) -> int:
    """Upserts scanned file rows (from NfsFilesystemSource/GlobusEndpointSource)
    into file_locations, resolving master_ref_id from images.base_name
    best-effort (a base_name can have no match if the raw/preview hasn't been
    ingested from blob yet). first_seen_at is set on first insert and then
    frozen - see the ON CONFLICT clause below."""
    if not rows:
        return 0

    base_name_to_master_ref_id = dict(
        conn.execute(
            "SELECT base_name, master_ref_id FROM images WHERE master_ref_id IS NOT NULL"
        ).fetchall()
    )
    conn.executemany(
        """
        INSERT INTO file_locations
            (base_name, extension, artifact_kind, storage_location, path, batch_label, sub_batch_index, master_ref_id, size_bytes, mtime_utc, scanned_at, first_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(storage_location, path) DO UPDATE SET
            base_name=excluded.base_name,
            extension=excluded.extension,
            artifact_kind=excluded.artifact_kind,
            batch_label=excluded.batch_label,
            sub_batch_index=excluded.sub_batch_index,
            master_ref_id=excluded.master_ref_id,
            size_bytes=excluded.size_bytes,
            mtime_utc=excluded.mtime_utc,
            scanned_at=excluded.scanned_at
        """,
        # first_seen_at is deliberately absent from the DO UPDATE SET above -
        # on conflict it keeps whatever value it was first inserted with,
        # even though this same scanned_at value is passed as the candidate
        # first_seen_at for every row (new or already-known).
        [
            (
                row["base_name"],
                row["extension"],
                row["artifact_kind"],
                row["storage_location"],
                row["path"],
                row.get("batch_label"),
                row.get("sub_batch_index"),
                base_name_to_master_ref_id.get(row["base_name"]),
                row.get("size_bytes"),
                row.get("mtime_utc"),
                row["scanned_at"],
                row["scanned_at"],
            )
            for row in rows
        ],
    )
    return len(rows)


def delete_stale_file_locations(conn, storage_location: str, current_paths: set) -> int:
    """Deletes file_locations rows for storage_location whose path isn't in
    current_paths (this run's full scan of that location). upsert_file_locations
    only inserts/updates - a file renamed or removed on disk since the last
    scan (e.g. a batch folder renamed to fix its location code) would
    otherwise leave a phantom row forever, still counting as raw_in_nfs=1 at
    its old path. Scoped to one storage_location per call so a source that
    was skipped this run (e.g. NFS temporarily unmounted, returning no
    entries) never gets its rows wiped out by an empty current_paths from a
    *different* source - see scan_file_locations.main, which only calls this
    for storage_locations that actually produced entries.

    Uses a temp table rather than a `path NOT IN (?, ?, ...)` bound-parameter
    list - a full NFS scan is 100k+ paths, well past SQLite's ~32766 host
    parameter limit ("too many SQL variables")."""
    if not current_paths:
        return 0
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _scanned_paths (path TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM _scanned_paths")
    conn.executemany("INSERT INTO _scanned_paths (path) VALUES (?)", [(p,) for p in current_paths])
    cursor = conn.execute(
        "DELETE FROM file_locations WHERE storage_location = ? "
        "AND path NOT IN (SELECT path FROM _scanned_paths)",
        (storage_location,),
    )
    conn.execute("DROP TABLE _scanned_paths")
    return cursor.rowcount
