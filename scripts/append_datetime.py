import concurrent.futures
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from utils.utils import (
    download_azcopy,
    find_most_recent_data_csv,
    get_exif_data,
    read_csv_as_df,
    read_yaml,
)

# Initialize logger
log = logging.getLogger(__name__)


class AppendDateTimeToTable:
    def __init__(self, cfg) -> None:

        self.keys = cfg.pipeline_keys
        self.csv_path = Path(
            find_most_recent_data_csv(
                Path(cfg.data.datadir, "processed_tables"), "merged_blobs_tables_metadata.csv"
                )
        )
        self.permanent_csv = Path(cfg.data.permanent_datadir,"merged_blobs_tables_metadata_permanent.csv")
        self.df = self.preprocess_df()
        
        self.config_keys()

    def preprocess_df(self):
        metadata_df = self.read_merged_table()
        metadata_permanent_df = self.load_reference_df()
        if isinstance(metadata_permanent_df, pd.DataFrame):
            # Mapping HasMatchingJpgAndRaw values from metadata_df to a dictionary for efficient lookup
            has_matching_dict = metadata_df.set_index('Name')['HasMatchingJpgAndRaw'].to_dict()
            # Update HasMatchingJpgAndRaw in metadata_permanent_df where names match
            metadata_permanent_df['HasMatchingJpgAndRaw'] = metadata_permanent_df['Name'].map(has_matching_dict).fillna(metadata_permanent_df['HasMatchingJpgAndRaw'])
            # Identify names not present in metadata_permanent_df
            names_not_in_permanent = set(metadata_df['Name']) - set(metadata_permanent_df['Name'])
            # Filter rows in metadata_df with names not in metadata_permanent_df
            rows_to_append = metadata_df[metadata_df['Name'].isin(names_not_in_permanent)]
            # Append these rows to metadata_permanent_df
            updated_metadata_permanent_df = pd.concat([metadata_permanent_df, rows_to_append], ignore_index=True)
            updated_metadata_permanent_df = self.fill_in_jpg_raw_values(updated_metadata_permanent_df)

            updated_metadata_permanent_df['CameraInfo_DateTime'] = pd.to_datetime(updated_metadata_permanent_df['CameraInfo_DateTime'], errors='coerce', format='%Y:%m:%d %H:%M:%S')
            # Assuming 'UsState' exists
            for index, row in updated_metadata_permanent_df.iterrows():
                if pd.isna(row['BatchID']) and not pd.isna(row['CameraInfo_DateTime']):
                # if pd.isna(row['BatchID']):
                    updated_metadata_permanent_df.at[index, 'BatchID'] = f"{row['UsState']}_{row['CameraInfo_DateTime'].strftime('%Y-%m-%d')}"
            # return updated_metadata_permanent_df
                elif pd.isna(row['BatchID']):
                    pass
            return updated_metadata_permanent_df
        else:
            return metadata_df
        
    def fill_in_jpg_raw_values(self, df):
        # First, ensure that df_jpg includes all necessary columns for the merge
        df_jpg_full = df[df['Name'].str.contains('.JPG')].copy()

        # Merge the original df (containing both ARW and JPG) with the df_jpg_full on Identifier, including necessary fields
        # This merge will ensure that for ARW rows, missing data is filled from corresponding JPG rows without losing any rows
        df_full_merged = df.merge(df_jpg_full[['BaseName', 'CameraInfo_DateTime', 'BatchID', 'SubBatchIndex']], 
                                on='BaseName', how='left', suffixes=('', '_from_jpg'))

        # For ARW rows, fill in missing values from JPG rows where applicable
        conditions = df_full_merged['Name'].str.contains('.ARW')
        df_full_merged.loc[conditions, 'CameraInfo_DateTime'] = df_full_merged.loc[conditions, 'CameraInfo_DateTime'].fillna(df_full_merged['CameraInfo_DateTime_from_jpg'])
        df_full_merged.loc[conditions, 'BatchID'] = df_full_merged.loc[conditions, 'BatchID'].fillna(df_full_merged['BatchID_from_jpg'])
        df_full_merged.loc[conditions, 'SubBatchIndex'] = df_full_merged.loc[conditions, 'SubBatchIndex'].fillna(df_full_merged['SubBatchIndex_from_jpg'])

        # Drop the temporary columns used for merging
        df_full_merged.drop(['CameraInfo_DateTime_from_jpg', 'BatchID_from_jpg', 'SubBatchIndex_from_jpg'], axis=1, inplace=True)
        return df_full_merged

    def load_reference_df(self):
        if Path(self.permanent_csv).exists():
            reference_df = read_csv_as_df(self.permanent_csv)
            return reference_df
        else:
            return False

    def read_merged_table(self):
        # Reads and preprocesses the CSV file at csv_path.
        
        log.info("Reading and preprocessing CSV data from %s.", self.csv_path)
        df = read_csv_as_df(self.csv_path)
        log.debug("CSV data read successfully.")
        return df

    def config_keys(self):
        # Configures keys from YAML configuration.
        yamkeys = read_yaml(self.keys)
        self.wir_sas_token = yamkeys["blobs"]["weedsimagerepo"]["sas_token"]
        self.wir_url = yamkeys["blobs"]["weedsimagerepo"]["url"]
        log.debug("WeedsImageRepo keys configured.")

    def get_jpg_df(self) -> pd.DataFrame:
        # Returns a DataFrame of JPGs missing from the specified stem list.
        batchid_none_df = self.df[self.df["Extension"] == "jpg"]
        batchid_none_df = batchid_none_df[pd.isnull(batchid_none_df['CameraInfo_DateTime'])]
        return batchid_none_df

    def download_jpg(self, imgurl: str, destination_path: str):
        azuresrc = imgurl + self.wir_sas_token
        # Assuming download_azcopy can accept a destination file path directly
        # If not, you may need to modify it to support downloading to a specific path
        log.debug("Downloading JPG for %s.", imgurl)
        download_azcopy(azuresrc, destination_path)

    def extract_exifdatetime(self, row: pd.Series) -> str:
        # Extracts the EXIF DateTimeOriginal from the JPG specified in the row.
        # Utilizes a temporary file to ensure automatic cleanup.
        # Use NamedTemporaryFile within a context manager to ensure cleanup
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_jpeg_path = tmp_file.name
            # Download the image directly to the temporary file
            self.download_jpg(row["ImageURL"], temp_jpeg_path)

        try:
            # Now that the file is closed, you can read the EXIF data
            exif_datetime = get_exif_data(temp_jpeg_path)["EXIF DateTimeOriginal"]
            log.debug("EXIF DateTimeOriginal extracted for %s.", row["Name"])
        finally:
            # Ensure the temporary file is removed after processing
            os.remove(temp_jpeg_path)

        return exif_datetime

    def update_dataframe_with_exif_data(self, missing_jpgs_df: pd.DataFrame) -> None:
        log.info("Updating DataFrame with EXIF DateTimeOriginal data.")

        def update_row(row):
            try:
                if 'CameraInfo_DateTime' in row and pd.isnull(row['CameraInfo_DateTime']):
                    exif_datetime = self.extract_exifdatetime(row)
                    return row["ImageURL"], exif_datetime
                else:
                    return row["ImageURL"], row['CameraInfo_DateTime']
            except Exception as e:
                log.error(
                    "Failed to update EXIF data for image %s: %s",
                    row["ImageURL"],
                    e,
                    exc_info=True,
                )
                return row["ImageURL"], None

        max_workers = int(len(os.sched_getaffinity(0)) / 3)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(update_row, row)
                for _, row in missing_jpgs_df.iterrows()
            ]
            for future in concurrent.futures.as_completed(futures):
                img_url, exif_datetime = future.result()
                if exif_datetime:
                    self.df.loc[
                        self.df["ImageURL"] == img_url, "CameraInfo_DateTime"
                    ] = exif_datetime
                    log.info("EXIF data updated for image: %s", img_url)

    def save_updated_dataframe(self) -> None:
        # Saves the updated DataFrame to a CSV file.
        try:
            self.df.to_csv(self.csv_path, index=False)
            self.df.to_csv(self.permanent_csv, index=False)
            log.info("DataFrame saved successfully to %s.", self.csv_path)
        except Exception as e:
            log.error(
                "Failed to save DataFrame to %s: %s",
                self.updated_csv_path,
                e,
                exc_info=True,
            )


def main(cfg: DictConfig) -> None:
    log.info("Main process started with task: %s", cfg.general.task)
    appenddatetime = AppendDateTimeToTable(cfg)
    df = appenddatetime.get_jpg_df()
    appenddatetime.update_dataframe_with_exif_data(df)
    appenddatetime.save_updated_dataframe()
