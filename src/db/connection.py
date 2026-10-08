import sqlite3
from pathlib import Path

# schema.sql's CREATE TABLE IF NOT EXISTS only creates a table on a brand-new
# DB - it's a no-op for a column added to a table that already exists on the
# live, populated field_exploration.db. Listed here so both a fresh DB (via
# schema.sql) and an existing one (via the ALTER TABLE below) end up with the
# same columns.
_COLUMN_MIGRATIONS = {
    "file_locations": ["sub_batch_index TEXT", "first_seen_at TEXT"],
    "file_status": [
        "sub_batch_index TEXT",
        "height TEXT",
        "size_class TEXT",
        "cotton_variety TEXT",
        "crop_type_secondary TEXT",
        "flower_fruit_or_seeds TEXT",
        "cloud_cover TEXT",
        "ground_residue TEXT",
        "ground_cover TEXT",
    ],
}


def _apply_column_migrations(conn: sqlite3.Connection) -> None:
    for table, columns in _COLUMN_MIGRATIONS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column in columns:
            name = column.split()[0]
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column}")

    # Rows upserted before first_seen_at existed have no real discovery time
    # on record. Backfill with a fixed pre-epoch sentinel rather than each
    # row's scanned_at: scanned_at is the same recent timestamp for every row
    # from one full-tree scan, so backfilling from it would make
    # NEWLY_PROCESSED_QUERY's first_seen_at >= cutoff briefly true for the
    # entire multi-year backlog right after this migration ships - a much
    # louder false signal than the staleness bug this column exists to fix.
    # A sentinel in the past means "existed before we tracked discovery time"
    # and can never look newly-processed for any real since_days window. This
    # is a no-op once every row has been backfilled.
    conn.execute(
        "UPDATE file_locations SET first_seen_at = '1970-01-01T00:00:00+00:00' WHERE first_seen_at IS NULL"
    )


def get_connection(db_path: str) -> sqlite3.Connection:
    """Opens a SQLite connection at db_path, creating the schema if needed."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    schema_path = Path(__file__).parent / "schema.sql"
    conn.executescript(schema_path.read_text())
    _apply_column_migrations(conn)
    return conn
