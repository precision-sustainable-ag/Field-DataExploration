import pandas as pd


def normalize_datetime(series: pd.Series) -> pd.Series:
    """Converts EXIF-style 'YYYY:MM:DD HH:MM:SS' timestamps to canonical
    'YYYY-MM-DD HH:MM:SS' text, leaving already-canonical values unchanged."""
    series = series.astype(str)
    series = series.str.replace(
        r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", regex=True
    )
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
