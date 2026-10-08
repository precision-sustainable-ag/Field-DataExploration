import re
from collections import Counter, defaultdict, namedtuple
from datetime import datetime, timezone

Location = namedtuple("Location", ["code", "display_name", "parent_code"])


def all_known_locations(conn) -> list:
    """Every location the DB knows about, with parent_code for callers that want
    to roll a child code (e.g. NC01) up into its parent (NC) instead of treating
    it as a separate location."""
    rows = conn.execute(
        "SELECT code, display_name, parent_code FROM locations ORDER BY code"
    ).fetchall()
    return [Location(*row) for row in rows]


def roll_up_to_parent(locations: list, code: str) -> str:
    """Maps a code to its parent_code if it has one, else returns it unchanged.
    Single explicit decision point for the NC01->NC-style rollup, replacing the
    three inconsistent ad hoc treatments across report.py/plot_by_season.py."""
    by_code = {location.code: location for location in locations}
    location = by_code.get(code)
    return location.parent_code if location and location.parent_code else code


def batch_folder_regex(locations: list) -> re.Pattern:
    """Builds the batch-folder regex (e.g. 'TX_2024-07-07') from the DB's actual
    location codes instead of a hand-written pattern that assumes every code is
    2 letters or 2 letters + 2 digits. A new location code of any shape works as
    soon as it's in the locations table."""
    codes = sorted({location.code for location in locations}, key=len, reverse=True)
    alternation = "|".join(re.escape(code) for code in codes)
    return re.compile(rf"^(?:{alternation})_\d{{4}}-\d{{2}}-\d{{2}}$")


def _majority_valid_prefix(rows: list, valid_codes: set) -> dict:
    """rows is a list of (key, base_name) pairs (key is whatever the caller
    wants corrected - a master_ref_id, a batch id, ...). Returns {key: code}
    for keys with an unambiguous majority 2-letter base_name prefix that's
    itself in valid_codes; a key with no valid-prefixed row, or a tie between
    two prefixes, is omitted (left for manual review rather than guessed)."""
    prefix_counts = defaultdict(Counter)
    for key, base_name in rows:
        prefix = base_name[:2].upper()
        if prefix in valid_codes:
            prefix_counts[key][prefix] += 1

    corrections = {}
    for key, counts in prefix_counts.items():
        ranked = counts.most_common()
        top_code, top_n = ranked[0]
        if len(ranked) > 1 and ranked[1][1] == top_n:
            continue
        corrections[key] = top_code
    return corrections


def infer_location_codes_from_images(conn, codes_to_fix: set, valid_codes: set) -> tuple:
    """For samples whose location_code is one of codes_to_fix (a known
    placeholder/typo, e.g. 'DV' entered instead of a real state), infers the
    true code from the 2-letter prefix of that sample's image base_names
    (e.g. 'OHA00623' -> 'OH') - that prefix is set by the field crew's camera
    naming convention, unlike the free-text UsState field that's sometimes
    typed wrong. Returns (corrections, unresolved): corrections is
    {master_ref_id: inferred_code}; unresolved is every other master_ref_id
    still on a to-fix code (see _majority_valid_prefix)."""
    placeholders = ",".join("?" * len(codes_to_fix))
    sample_ids = {
        row[0]
        for row in conn.execute(
            f"SELECT master_ref_id FROM samples WHERE location_code IN ({placeholders})",
            tuple(codes_to_fix),
        ).fetchall()
    }
    if not sample_ids:
        return {}, []

    rows = conn.execute(
        f"""
        SELECT s.master_ref_id, i.base_name
        FROM samples s JOIN images i ON i.master_ref_id = s.master_ref_id
        WHERE s.location_code IN ({placeholders}) AND i.base_name IS NOT NULL
        """,
        tuple(codes_to_fix),
    ).fetchall()

    corrections = _majority_valid_prefix(rows, valid_codes)
    unresolved = sorted(sample_ids - set(corrections))
    return corrections, unresolved


def infer_batch_location_codes_from_images(conn, codes_to_fix: set, valid_codes: set) -> tuple:
    """Same idea as infer_location_codes_from_images, but for batches.location_code
    (parsed from the batch_label prefix at creation time, e.g. 'DV_2025-06-11' -
    an independent copy of the same UsState typo, not derived from samples).
    Only corrects the DB column; does not touch batch_label or any physical
    NFS/JUNO folder name, since those need a real rename, not just a row
    update - see any batch this flags as corrected before renaming its
    on-disk folder to match. Returns (corrections, unresolved) keyed by
    batches.id."""
    placeholders = ",".join("?" * len(codes_to_fix))
    batch_ids = {
        row[0]
        for row in conn.execute(
            f"SELECT id FROM batches WHERE location_code IN ({placeholders})",
            tuple(codes_to_fix),
        ).fetchall()
    }
    if not batch_ids:
        return {}, []

    rows = conn.execute(
        f"""
        SELECT b.id, i.base_name
        FROM batches b JOIN images i ON i.batch_id = b.id
        WHERE b.location_code IN ({placeholders}) AND i.base_name IS NOT NULL
        """,
        tuple(codes_to_fix),
    ).fetchall()

    corrections = _majority_valid_prefix(rows, valid_codes)
    unresolved = sorted(batch_ids - set(corrections))
    return corrections, unresolved


def _record_corrections(conn, entity_type: str, table: str, id_column: str, corrections: dict) -> None:
    """Inserts one location_code_corrections row per entity about to be
    corrected, capturing its current (about-to-be-overwritten) value as
    previous_code - must run before the UPDATE that applies `corrections`,
    not after, or previous_code would just be the new value."""
    if not corrections:
        return
    placeholders = ",".join("?" * len(corrections))
    previous_by_id = dict(
        conn.execute(
            f"SELECT {id_column}, location_code FROM {table} WHERE {id_column} IN ({placeholders})",
            tuple(corrections.keys()),
        ).fetchall()
    )
    corrected_at = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO location_code_corrections
            (entity_type, entity_id, previous_code, corrected_code, corrected_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (entity_type, str(entity_id), previous_by_id.get(entity_id), new_code, corrected_at)
            for entity_id, new_code in corrections.items()
        ],
    )


def apply_location_corrections(conn, corrections: dict) -> int:
    """Overwrites samples.location_code for the given {master_ref_id: code}
    map, recording each change in location_code_corrections first. Used to
    write back the result of infer_location_codes_from_images."""
    if not corrections:
        return 0
    _record_corrections(conn, "sample", "samples", "master_ref_id", corrections)
    conn.executemany(
        "UPDATE samples SET location_code = ? WHERE master_ref_id = ?",
        [(code, master_ref_id) for master_ref_id, code in corrections.items()],
    )
    return len(corrections)


def apply_batch_location_corrections(conn, corrections: dict) -> int:
    """Overwrites batches.location_code for the given {batch_id: code} map,
    recording each change in location_code_corrections first. Used to write
    back the result of infer_batch_location_codes_from_images."""
    if not corrections:
        return 0
    _record_corrections(conn, "batch", "batches", "id", corrections)
    conn.executemany(
        "UPDATE batches SET location_code = ? WHERE id = ?",
        [(code, batch_id) for batch_id, code in corrections.items()],
    )
    return len(corrections)
