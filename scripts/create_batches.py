import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from azure.storage.blob import BlobServiceClient
from omegaconf import DictConfig
from tqdm import tqdm

from utils.utils import find_most_recent_csv, read_yaml

log = logging.getLogger(__name__)


class FieldBatchLister:
    
    def __init__(self, cfg):
        """Initializes the lister with configurations to manage blob storage operations."""
        self.keys = self.config_keys(cfg.pipeline_keys)
        self.timestamp = datetime.now().strftime("%Y%m%d")
        all_blobs = self.list_unique_folders()
        self.df = self.get_blobs_per_batch(all_blobs)

    def config_keys(self, keypath):
        """Reads and configures keys from the YAML configuration for Azure Blob access."""
        yamkeys = read_yaml(keypath)
        self.field_batches_sas_token = yamkeys["blobs"]["field-batches"]["read_sas_token"]
        self.field_batches_url = yamkeys["blobs"]["account_url"]
        self.container_name = "field-batches"
        log.debug("WeedsImageRepo keys configured.")

    def list_unique_folders(self):
        """
        List all unique folders in a given container using the Azure Blob Service client.
        Filters folders based on a predefined pattern.
        """
        blob_service_client = BlobServiceClient(account_url=self.field_batches_url, credential=self.field_batches_sas_token)
        container_client = blob_service_client.get_container_client(self.container_name)
        all_blobs = []
        for blob in tqdm(container_client.list_blobs()):
            blob_name = blob.name
            if ("raws" in blob_name) and not ("preprocessed" in blob_name):
                all_blobs.append(blob_name)
        return all_blobs
    
    def get_blobs_per_batch(self, all_blobs):
        """Organizes blobs into batches for processing, categorizing by directory structure."""
        df = pd.DataFrame(all_blobs, columns=['BlobName'])
        df[['BatchID', 'Subfolder1', 'Subfolder2', 'FileName']] = df['BlobName'].str.split('/',n=4, expand=True)
        df["BaseName"] = df["FileName"].str.rsplit(".", n=1).str[0]
        df = df.sort_values(by=["BatchID"])
        df["BatchFolder"] = df["BatchID"] + "/" + df["Subfolder1"] + "/" + df["Subfolder2"]
        return df


def round_down_to_nearest_3_hours(dt: datetime) -> datetime:
    # Rounds down a datetime object to the nearest 3-hour block.
    rounded_hour = dt.hour - (dt.hour % 3)
    return dt.replace(hour=rounded_hour, minute=0, second=0, microsecond=0)


class CreateBatchProcessor:
    """Processor for handling batch operations on images based on metadata and Azure Blob storage."""
    def __init__(self, cfg: DictConfig) -> None:
        self.ykeys = read_yaml(cfg.pipeline_keys)
        self.datadir = Path(cfg.paths.datadir, "processed_tables")
        self.csv_path = Path(find_most_recent_csv(self.datadir, "merged_blobs_tables_metadata.csv"))
        self.permanent_csv = Path(cfg.paths.persistent_datadir,"merged_blobs_tables_metadata_permanent.csv")
        self.file_path = "./tempoutputfieldbatches.txt"
        self.read_and_convert_datetime()

    def config_keys(self) -> None:
        """Configures Azure Blob Storage access for both read and write operations."""
        self.write_fbatch_key = self.ykeys["blobs"]["field-batches"]["write_sas_token"]
        self.fbatch_url = self.ykeys["blobs"]["field-batches"]["url"]
        self.read_weedimgrepo_key = self.ykeys["blobs"]["weedsimagerepo"]["sas_token"]
        self.weedimgrepo_url = self.ykeys["blobs"]["weedsimagerepo"]["url"]
        log.debug("Configured keys for image repository and field batches.")
    
    def replace_date_format(self, date: Optional[str]) -> Optional[str]:
        """Defines the regex pattern to match the date format and replaces ':' with '-'."""
        pattern = r'(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})'
        if pd.isna(date):
            return date
        return re.sub(pattern, r'\1-\2-\3 \4:\5:\6', date)
    
    def standardize_dt_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the date format standardization to the DataFrame."""
        df['CameraInfo_DateTime'] = df['CameraInfo_DateTime'].apply(self.replace_date_format)
        df['CameraInfo_DateTime'] = pd.to_datetime(df['CameraInfo_DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        return df

    def read_and_convert_datetime(self) -> pd.DataFrame:
        """Reads and converts datetime columns in the CSV to standardized datetime format.
           Use the data table that has image capture datetime info."""
        self.df = pd.read_csv(self.csv_path, dtype={'CameraInfo_DateTime': str})
        if "CameraInfo_DateTime" not in self.df.columns:
            self.df = pd.read_csv(self.permanent_csv, dtype={'CameraInfo_DateTime': str})
        self.df = self.standardize_dt_format(self.df)

    def split_datetime(self) -> "CreateBatchProcessor":
        """Splits datetime into separate date and time columns for further processing."""
        self.df["CameraInfo_Date"] = self.df["CameraInfo_DateTime"].dt.date
        self.df["CameraInfo_Time"] = self.df["CameraInfo_DateTime"].dt.time
        return self

    def preprocess_df(self) -> "CreateBatchProcessor":
        # Filters and cleans the DataFrame based on matching JPG and RAWs, and JPG extension.
        log.info("Preprocessing DataFrame")
        self.df = self.df[self.df["HasMatchingJpgAndRaw"] == True].dropna(subset=["UsState"])
        self.df = self.df[self.df["Extension"] == "jpg"]
        self.df = self.df.dropna(subset="CameraInfo_DateTime")
        return self

    def adjust_groups(self) -> pd.DataFrame:
        """Adjusts DataFrame to group data for batch processing."""
        log.info("Adjusting groups for batch processing")
        self.df = self.df.sort_values(by=['UsState', 'MasterRefID', 'CameraInfo_DateTime'])
        # Apply this rounding to the CameraInfo_DateTime column to create a 3-hourly grouping column
        self.df['ThreeHourlyGroup'] = self.df['CameraInfo_DateTime'].apply(round_down_to_nearest_3_hours)
        self.df['SubBatchIndex'] = self.df.sort_values(by=['ThreeHourlyGroup']).groupby(['UsState', 'CameraInfo_Date'])['ThreeHourlyGroup'].transform(lambda x: pd.factorize(x)[0] + 1)
        self.df['SubBatchIndex_Padded'] = self.df['SubBatchIndex'].apply(lambda x: f"{x:0{2}d}")
        self.df =  self.df.sort_values(by=["UsState", "CameraInfo_Date", "SubBatchIndex"])
        self.df['batches'] = self.df.apply(lambda row: f"{row['UsState']}_{row['CameraInfo_Date'].strftime('%Y-%m-%d')}/raws/{row['SubBatchIndex_Padded']}/{row['Name'].replace('JPG', 'ARW')}", axis=1)
        return self

    def add_extra_number(self, cell_value):
        """Increments the last number in the provided string by 1 after padding with zeros."""
        last_number = int(cell_value[-1])
        new_last_number = str(last_number + 1).zfill(len(cell_value))
        return cell_value[:-1] + new_last_number
    
    def filter_batched_data(self,present_batches_df):
        """Filters out already processed batches from the DataFrame and handles duplicate batches."""
        self.df = self.df[~self.df["BaseName"].isin(present_batches_df["BaseName"])]
        if len(self.df) == 0:
            log.info("No new images present. No images to be moved to the field-batches blob containers. Exiting.")
            exit(0)

        self.df[["BatchID_y", "Subfolder1", "Subfolder2", "FName"]] = self.df['batches'].str.split('/', expand=True)
        self.df["BatchFolder"] = self.df["BatchID_y"] + "/" + self.df["Subfolder1"] + "/" + self.df["Subfolder2"]
        
        duplicate_batch_folders = self.df[self.df["BatchFolder"].isin(present_batches_df["BatchFolder"])]
        if not duplicate_batch_folders.empty:
            log.warning("Duplicate batch folders. Renaming the duplicates. Duplicates include...")
            for index, row in duplicate_batch_folders.iterrows():
                log.warning(f'{row["BatchFolder"]}/{row["FName"]}')
            log.warning("Consider investiagting duplicate folders. Exiting.")
            exit(0)
            
    def move_from_weeedsimagerepo2fieldbatches(self, batch: str) -> None:
        """Moves batches from the weeds image repository to field batches using azcopy."""
        log.info(f"Moving batch {batch} from weeds image repository to field batches")
        blob_name = Path(batch).name
        weedimgrepo_src = f"{self.weedimgrepo_url}/{blob_name}{self.read_weedimgrepo_key}"
        fieldbatch_dst = f"{self.fbatch_url}/{batch}{self.write_fbatch_key}"

        command = f'azcopy copy "{weedimgrepo_src}" "{fieldbatch_dst}" --recursive'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            log.info("Copy successful")
            log.debug(result.stdout)
        else:
            log.error("Error in copy operation")
            log.debug(result.stderr)

    def process_df(self) -> None:
        """Processes the DataFrame without using concurrency, processing each batch one at a time."""
        log.info("Processing DataFrame without concurrency")
        for _, row in self.df.reset_index().iterrows():
            batch = row["batches"]
            self.move_from_weeedsimagerepo2fieldbatches(batch)

    def process_df_concurrently(self) -> None:
        """Processes the DataFrame using concurrency to handle multiple batches simultaneously."""
        log.info("Processing DataFrame with concurrency")
        batches = self.df["batches"].unique()
        max_workers = int(len(os.sched_getaffinity(0)) / 3)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.move_from_weeedsimagerepo2fieldbatches, batch) for batch in batches]
            for future in as_completed(futures):
                try:
                    future.result()  # If the function returns something you can capture it here
                except Exception as e:
                    log.error(f"Error processing batch: {e}")


def main(cfg: DictConfig) -> None:
    """Main function to orchestrate the batch processing based on configurations provided."""
    log.info(f"Starting {cfg.general.task}")
    
    # Initialize the FieldBatchLister to fetch and list current batch data
    present_batches_df = FieldBatchLister(cfg).df
    present_batches_df.to_csv("present_batches.csv", index=False)
    
    # Set up the batch processor and perform initial configurations
    dataproc = CreateBatchProcessor(cfg)
    dataproc.config_keys()
    dataproc.split_datetime()
    dataproc.preprocess_df()
    dataproc.adjust_groups()
    dataproc.filter_batched_data(present_batches_df)
    
    # Choose the method of processing: concurrently or sequentially
    run_concurrent = True
    if run_concurrent:
        dataproc.process_df_concurrently()
    else:
        dataproc.process_df()

    log.info(f"Task '{cfg.general.task}' completed successfully.")
