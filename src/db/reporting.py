import pandas as pd

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
