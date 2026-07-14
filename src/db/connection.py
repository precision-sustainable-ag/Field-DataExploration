import sqlite3
from pathlib import Path


def get_connection(db_path: str) -> sqlite3.Connection:
    """Opens a SQLite connection at db_path, creating the schema if needed."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    schema_path = Path(__file__).parent / "schema.sql"
    conn.executescript(schema_path.read_text())
    return conn
