import re
from collections import namedtuple

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
