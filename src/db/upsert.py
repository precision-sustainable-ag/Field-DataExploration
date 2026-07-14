import json
import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def _split_name(blob_name: str) -> tuple:
    base_name, _, extension = blob_name.rpartition(".")
    return (base_name or blob_name), extension.lower()


def build_locations(codes: set) -> list:
    """Infers parent_code for codes like NC01 -> NC (parent must also be in codes)."""
    codes = sorted(codes)
    rows = []
    for code in codes:
        match = re.match(r"^([A-Z]{2})\d+$", code)
        parent_code = match.group(1) if match and match.group(1) in codes else None
        rows.append((code, code, parent_code))
    return rows


def upsert_locations(conn, state_list: list, codes_in_data: set) -> None:
    missing_from_config = codes_in_data - set(state_list)
    if missing_from_config:
        log.warning(
            f"Location codes present in data but missing from cfg.state_list, "
            f"adding them to locations anyway: {sorted(missing_from_config)}"
        )

    rows = build_locations(set(state_list) | codes_in_data)
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
