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
    convert_datetime
)

# Initialize logger
log = logging.getLogger(__name__)

class CameraInfoUpdater:
    def __init__(self, df, extensions):
        self.extensions = extensions
        self.df = df
        self.nan_caminfo_rows = None
        self.jpg_pers = None

    def load_and_clean_data(self):
        """Load the CSV file and clean the 'Name' column by removing extensions."""
        pattern = '|'.join(self.extensions)
        self.df['Stem'] = self.df['Name'].str.replace(pattern, '', case=False, regex=True)

    def separate_rows_by_nan(self, column):
        """Separate rows into those with and without NaN in the specified column."""
        not_nan_rows = self.df[~self.df[column].isna()].copy()
        self.nan_caminfo_rows = self.df[self.df[column].isna()].copy()

    def filter_jpg_images(self):
        """Filter the DataFrame for JPEG images that match the given stems."""
        missing_caminfo_stems = self.nan_caminfo_rows["Stem"]
        jpg_df = self.df[self.df["Extension"].str.lower() == 'jpg'].copy()
        self.jpg_pers = jpg_df[jpg_df["Stem"].isin(missing_caminfo_stems)].copy()

    def merge_camera_info(self):
        """Merge JPEG rows with missing CameraInfo rows based on the 'Stem' column."""
        return self.nan_caminfo_rows.merge(self.jpg_pers[['Stem', 'CameraInfo_DateTime']], on='Stem', how='left')

    def update_camera_info(self, merged_df):
        """Update the 'CameraInfo_DateTime' in nan_caminfo_rows with values from merged_df."""
        nan_caminfo_rows_copy = self.nan_caminfo_rows.copy()
        nan_caminfo_rows_copy.loc[:, 'CameraInfo_DateTime'] = merged_df['CameraInfo_DateTime_y'].values
        return nan_caminfo_rows_copy

    def update_original_df(self, updated_nan_caminfo_rows):
        """Update the original DataFrame with CameraInfo_DateTime from updated_nan_caminfo_rows."""
        df_updated = self.df.copy()
        # df_updated.set_index('Stem', inplace=True)
        # updated_nan_caminfo_rows.set_index('Stem', inplace=True)
        df_updated.update(updated_nan_caminfo_rows[['CameraInfo_DateTime']])
        df_updated = df_updated.drop(columns=['index'])
        df_updated.reset_index(inplace=True)
        return df_updated


    def verify_updates(self, df_updated, original_nan_count, updated_nan_caminfo_rows):
        """Verify that the updates were applied correctly."""
        updated_nan_count = df_updated['CameraInfo_DateTime'].isna().sum()
        print(f"Original NaN count: {original_nan_count}")
        print(f"Updated NaN count: {updated_nan_count}")
        
        # Verify a few specific rows that were supposed to be updated
        updated_rows = df_updated[df_updated['Stem'].isin(updated_nan_caminfo_rows.index)]
        print("Sample of updated rows:")
        print(updated_rows.head())
        
        # Additional check: how many NaN values were filled
        filled_nan_count = original_nan_count - updated_nan_count
        print(f"Number of NaN values filled: {filled_nan_count}")

    def run(self):
        # Load and clean data
        self.load_and_clean_data()
        
        # Separate rows by NaN in 'CameraInfo_DateTime'
        self.separate_rows_by_nan('CameraInfo_DateTime')
        
        # Filter JPEG images
        self.filter_jpg_images()
        
        # Merge DataFrames
        merged_df = self.merge_camera_info()
        
        # Update CameraInfo in nan_caminfo_rows
        updated_nan_caminfo_rows = self.update_camera_info(merged_df)
        
        # Update the original DataFrame
        original_nan_count = self.df['CameraInfo_DateTime'].isna().sum()
        df_updated = self.update_original_df(updated_nan_caminfo_rows)
        
        # Verify updates
        self.verify_updates(df_updated, original_nan_count, updated_nan_caminfo_rows)
        return df_updated
    
class AppendDateTimeToTable:
    """Manages the process of appending date-time metadata to a CSV table from JPG EXIF data."""
    def __init__(self, cfg) -> None:

        self.keys = cfg.pipeline_keys
        self.csv_path = Path(find_most_recent_data_csv(Path(cfg.data.datadir, "processed_tables"), "merged_blobs_tables_metadata.csv"))
        self.permanent_csv = Path(cfg.data.persistent_datadir,"merged_blobs_tables_metadata_permanent.csv")
        self.df = self.preprocess_df()
        self.config_keys()

    def preprocess_df(self) -> pd.DataFrame:
        """Processes the DataFrame by merging new data and updating fields."""
        metadata_df = self.read_merged_table()
        metadata_permanent_df = self.load_reference_df()
        
        if isinstance(metadata_permanent_df, pd.DataFrame):
            self.update_has_matching(metadata_df, metadata_permanent_df)
            rows_to_append = self.identify_new_data(metadata_df, metadata_permanent_df)
            if rows_to_append.empty:
                log.info("No new data to add to the persistent data table.")
                return metadata_permanent_df  # If no new data, return the unchanged permanent DataFrame
            updated_metadata_permanent_df = self.append_new_data(metadata_permanent_df, rows_to_append)
            updated_metadata_permanent_df = self.fill_in_jpg_raw_values(updated_metadata_permanent_df)
            updated_metadata_permanent_df = self.update_date_time(updated_metadata_permanent_df)
            return updated_metadata_permanent_df
        else:
            return metadata_df
        
    def update_has_matching(self, metadata_df: pd.DataFrame, metadata_permanent_df: pd.DataFrame) -> None:
        """Updates 'HasMatchingJpgAndRaw' field in the permanent DataFrame based on the latest data."""
        has_matching_dict = metadata_df.set_index('Name')['HasMatchingJpgAndRaw'].to_dict()
        metadata_permanent_df['HasMatchingJpgAndRaw'] = metadata_permanent_df['Name'].map(has_matching_dict).fillna(metadata_permanent_df['HasMatchingJpgAndRaw'])

    def identify_new_data(self, metadata_df: pd.DataFrame, metadata_permanent_df: pd.DataFrame) -> pd.DataFrame:
        """Identifies new rows in metadata DataFrame that are not present in the permanent DataFrame."""
        names_not_in_permanent = set(metadata_df['Name']) - set(metadata_permanent_df['Name'])
        return metadata_df[metadata_df['Name'].isin(names_not_in_permanent)]

    def append_new_data(self, metadata_permanent_df: pd.DataFrame, rows_to_append: pd.DataFrame) -> pd.DataFrame:
        """Appends new rows to the permanent DataFrame."""
        return pd.concat([metadata_permanent_df, rows_to_append], ignore_index=True)

    def update_date_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Updates the 'CameraInfo_DateTime' and 'BatchID' fields based on 'CameraInfo_DateTime'."""
        df['CameraInfo_DateTime'] = pd.to_datetime(df['CameraInfo_DateTime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
        for index, row in df.iterrows():
            if pd.isna(row['BatchID']) and not pd.isna(row['CameraInfo_DateTime']):
                df.at[index, 'BatchID'] = f"{row['UsState']}_{row['CameraInfo_DateTime'].strftime('%Y-%m-%d')}"
            elif pd.isna(row['BatchID']):
                pass
        return df
        
    def fill_in_jpg_raw_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values in ARW rows in the DataFrame by merging and filling in corresponding data from the JPG row."""
        # Use the CameraInfoUpdater to handle the filling in process
        camera_info_updater = CameraInfoUpdater(df=df, extensions=['.jpg', '.arw', '.ARW', '.JPG'])
        df_updated = camera_info_updater.run()
        return df_updated

    def load_reference_df(self) -> pd.DataFrame:
        """Loads the reference DataFrame from permanent storage."""
        if Path(self.permanent_csv).exists():
            reference_df = read_csv_as_df(self.permanent_csv)
            return reference_df
        else:
            return False

    def read_merged_table(self):
        """Reads and preprocesses the CSV file specified by csv_path."""
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
        """ Download image using image url and AZCOPY"""
        azuresrc = imgurl + self.wir_sas_token
        log.debug("Downloading JPG for %s.", imgurl)
        download_azcopy(azuresrc, destination_path)

    def extract_exifdatetime(self, row: pd.Series) -> str:
        """
        Extracts the EXIF DateTimeOriginal from the JPG specified in the row using a temporary file.

        Args:
            row (pd.Series): A data series containing the image URL and other metadata.

        Returns:
            str: The extracted EXIF DateTimeOriginal, or None if extraction fails.
        """
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
        """
        Updates the DataFrame with EXIF DateTimeOriginal data extracted from images.

        Args:
            missing_jpgs_df (pd.DataFrame): DataFrame containing records of JPG images missing EXIF date-time.
        """
        log.info("Updating DataFrame with EXIF DateTimeOriginal data.")
        
        # Local function to process each image and update its EXIF data.
        def update_row(row):
            """Process an individual row to extract and update EXIF DateTime data."""
            try:
                # Proceed only if the CameraInfo_DateTime is missing.
                if 'CameraInfo_DateTime' in row and pd.isnull(row['CameraInfo_DateTime']):
                    exif_datetime = self.extract_exifdatetime(row)
                    return row["ImageURL"], exif_datetime
                else:
                    return row["ImageURL"], row['CameraInfo_DateTime']
            except Exception as e:
                log.error("Failed to update EXIF data for image %s: %s", row["ImageURL"], e, exc_info=True)
                return row["ImageURL"], None

        # Determine the number of workers based on available CPU cores, reducing potential over-utilization.
        max_workers = int(len(os.sched_getaffinity(0)) / 3)
        log.info(f"Number of JPGs missing exif info: {missing_jpgs_df.shape[0]}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create futures for concurrent execution.
            futures = [
                executor.submit(update_row, row)
                for _, row in missing_jpgs_df.iterrows()
            ]
            # Process the results as they complete.
            for future in concurrent.futures.as_completed(futures):
                img_url, exif_datetime = future.result()
                if exif_datetime:
                    # Update the DataFrame in place if new EXIF data is obtained.
                    self.df.loc[self.df["ImageURL"] == img_url, "CameraInfo_DateTime"] = exif_datetime
                    log.info("EXIF data updated for image: %s", img_url)

    def save_updated_dataframe(self) -> None:
        # Saves the updated DataFrame to a CSV file.
        try:
            self.df['CameraInfo_DateTime'] = self.df['CameraInfo_DateTime'].apply(convert_datetime)
            self.df.to_csv(self.csv_path, index=False)
            log.info("Today's DataFrame saved successfully to %s.", self.csv_path)
            self.df.to_csv(self.permanent_csv, index=False)
            log.info("Persistent data DataFrame saved successfully to %s.", self.permanent_csv)
        except Exception as e:
            log.error("Failed to save DataFrame to %s: %s", self.csv_path, e, exc_info=True)


def main(cfg: DictConfig) -> None:
    log.info("Main process started with task: %s", cfg.general.task)
    appenddatetime = AppendDateTimeToTable(cfg)
    df = appenddatetime.get_jpg_df()
    appenddatetime.update_dataframe_with_exif_data(df)
    appenddatetime.save_updated_dataframe()
    log.info("Main process ended with task: %s", cfg.general.task)
