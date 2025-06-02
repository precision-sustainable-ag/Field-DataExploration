import logging
import os
import re
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from omegaconf import DictConfig

from utils.utils import (
    download_azcopy, find_most_recent_data_csv, get_exif_data,
    read_csv_as_df, read_yaml, convert_datetime
)

log = logging.getLogger(__name__)

class CameraInfoHelper:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def fill_missing_by_stem(self) -> pd.DataFrame:
        """Fills missing CameraInfo_DateTime in ARWs using matching JPGs by stem."""
        jpg_lookup = self.df[
            self.df['CameraInfo_DateTime'].notna() & (self.df['Extension'].str.lower() == 'jpg')
        ][['Stem', 'CameraInfo_DateTime']].drop_duplicates()
        
        df_updated = self.df.merge(
            jpg_lookup, on='Stem', how='left', suffixes=('', '_jpg')
        )
        df_updated['CameraInfo_DateTime'] = df_updated['CameraInfo_DateTime'].fillna(df_updated['CameraInfo_DateTime_jpg'])
        return df_updated.drop(columns=['CameraInfo_DateTime_jpg'])

    def update_with_downloaded_exif(self, sas_token: str, wir_url: str) -> pd.DataFrame:
        """Update rows with missing CameraInfo_DateTime by downloading EXIF info from image URLs."""
        def download_and_extract(row):
            try:
                if pd.notna(row["CameraInfo_DateTime"]):
                    return None

                img_url = row["ImageURL"] + sas_token
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    download_azcopy(img_url, tmp_file.name)
                exif = get_exif_data(tmp_file.name)
                os.remove(tmp_file.name)
                return row["ImageURL"], exif.get("EXIF DateTimeOriginal")
            except Exception as e:
                log.warning("EXIF extraction failed for %s: %s", row["ImageURL"], str(e))
                return row["ImageURL"], None

        missing_rows = self.df[
            (self.df['Extension'].str.lower() == 'jpg') & (self.df['CameraInfo_DateTime'].isna())
        ]

        max_workers = max(1, int(len(os.sched_getaffinity(0)) / 3))
        log.info("Processing %d JPGs for EXIF recovery...", len(missing_rows))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_and_extract, row) for _, row in missing_rows.iterrows()]
            for future in as_completed(futures):
                url, exif_dt = future.result()
                if exif_dt:
                    self.df.loc[self.df["ImageURL"] == url, "CameraInfo_DateTime"] = exif_dt

        return self.df

def normalize_datetime_column(series: pd.Series) -> pd.Series:
    """
    Normalize a datetime column that may contain a mix of ISO and EXIF formats.

    - EXIF format: "YYYY:MM:DD HH:MM:SS"
    - ISO format: "YYYY-MM-DD HH:MM:SS"
    """
    # Convert to string and replace EXIF-style ":" with "-" only in the date part
    series = series.astype(str)
    series = series.str.replace(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", regex=True)

    return pd.to_datetime(series, errors='coerce')

class EXIFMetadataManager:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.csv_path = Path(find_most_recent_data_csv(Path(cfg.paths.datadir, "processed_tables"), "merged_blobs_tables_metadata.csv"))
        self.permanent_csv = Path(cfg.paths.persistent_datadir, "merged_blobs_tables_metadata_permanent.csv")
        self.lts_csv = Path(cfg.paths.longterm_storage).parent / "field-tools" / "persistent_data_tables" / "merged_blobs_tables_metadata_lts.csv"
        self.df = read_csv_as_df(self.csv_path)
        self.keys = read_yaml(cfg.pipeline_keys)
        self.sas_token = self.keys["blobs"]["weedsimagerepo"]["sas_token"]
        self.wir_url = self.keys["blobs"]["weedsimagerepo"]["url"]

    def load_and_prepare_dataframe(self) -> pd.DataFrame:
        """Merge in permanent data and prepare the working DataFrame."""
        log.debug(f"Loaded CSV from {self.csv_path} with shape: {self.df.shape}")
        ref_df = read_csv_as_df(self.permanent_csv) if self.permanent_csv.exists() else None
        log.debug(f"Loaded permanent CSV from {self.permanent_csv} with shape: {ref_df.shape if ref_df is not None else 'None'}")
        if ref_df is not None:
            new_data = self.df[~self.df['Name'].isin(ref_df['Name'])]
            updated_df = pd.concat([ref_df, new_data], ignore_index=True)
        else:
            updated_df = self.df

        updated_df['Stem'] = updated_df['Name'].str.replace(r'\.(jpg|arw)$', '', case=False, regex=True)

        # Create the extension column again
        updated_df['Extension'] = updated_df['Name'].str.extract(r'\.(jpg|arw|JPG|ARW)$', flags=re.IGNORECASE)[0].str.lower()
        
        log.debug(f"Updated Perrmanent DataFrame shape: {updated_df.shape}")

        log.info(f"Total new rows added: {self.df.shape[0] - ref_df.shape[0]}")
        return updated_df

    def update_missing_camera_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills in missing CameraInfo using both JPG stem lookup and downloaded EXIF."""
        helper = CameraInfoHelper(df)
        df_filled = helper.fill_missing_by_stem()
        helper.df = df_filled
        df = helper.update_with_downloaded_exif(sas_token=self.sas_token, wir_url=self.wir_url)
        df['CameraInfo_DateTime'] = normalize_datetime_column(df['CameraInfo_DateTime'])
        df['BatchID'] = df.apply(
            lambda r: f"{r['UsState']}_{r['CameraInfo_DateTime'].strftime('%Y-%m-%d')}" if pd.isna(r['BatchID']) and pd.notna(r['CameraInfo_DateTime']) else r['BatchID'], axis=1
        )
        return df

    def save_updated_dataframes(self, df: pd.DataFrame) -> None:
        """Saves updated DataFrame to all configured paths."""
        df.to_csv(self.csv_path, index=False)
        df.to_csv(str(self.permanent_csv), index=False)
        log.info("Saved updated data to both %s and %s", self.csv_path, self.permanent_csv)
        log.info(f"Final DataFrame shape: {df.shape}")
        
        # df.to_csv(self.lts_csv, index=False)
        # log.info("Long-term table updated at %s", self.lts_csv)


def main(cfg: DictConfig) -> None:
    log.info("Starting CameraInfo_DateTime update pipeline.")
    manager = EXIFMetadataManager(cfg)
    df_prepared = manager.load_and_prepare_dataframe()
    df_updated = manager.update_missing_camera_info(df_prepared)
    manager.save_updated_dataframes(df_updated)
    log.info("Pipeline completed.")

if __name__ == "__main__":
    main()