import datetime as dt
import logging
import os
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Set

import pandas as pd
from azure.storage.blob import BlobServiceClient
from dateutil.parser import parse
from omegaconf import DictConfig
from tqdm import tqdm

from utils.utils import azcopy_list, find_most_recent_csv, read_csv_as_df, read_yaml

log = logging.getLogger(__name__)


class FieldBatchLister:
    
    def __init__(self, cfg):
        """
        Initializes the comparer with configurations, container names, SAS tokens, and output directory.
        """
        self.keys = self.config_keys(cfg.pipeline_keys)

        # self.output_dir = Path(cfg.general.field_results)
        # self.output_dir.mkdir(exist_ok=True, parents=True)

        self.timestamp = datetime.now().strftime("%Y%m%d")

        all_blobs = self.list_unique_folders()
        self.df = self.get_blobs_per_batch(all_blobs)

    def config_keys(self, keypath):
        # Configures keys from YAML configuration.
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
        blob_service_client = BlobServiceClient(
            account_url=self.field_batches_url, credential=self.field_batches_sas_token
        )
        container_client = blob_service_client.get_container_client(self.container_name)
        all_blobs = []
        for blob in tqdm(container_client.list_blobs()):
        # for blob in tqdm(container_client.walk_blobs(delimiter=".")):
            blob_name = blob.name
            if ("raws" in blob_name) and not ("preprocessed" in blob_name):
                all_blobs.append(blob_name)
        
        return all_blobs
    
    def get_blobs_per_batch(self, all_blobs):
        df = pd.DataFrame(all_blobs, columns=['BlobName'])
        df[['BatchID', 'Subfolder1', 'Subfolder2', 'FileName']] = df['BlobName'].str.split('/',n=4, expand=True)
        df["BaseName"] = df["FileName"].str.rsplit(".", n=1).str[0]
        df = df.sort_values(by=["BatchID"])
        df["Batched"] = True
        df["BatchFolder"] = df["BatchID"] + "/" + df["Subfolder1"] + "/" + df["Subfolder2"]
        return df


def round_down_to_nearest_3_hours(dt: datetime) -> datetime:
    # Rounds down a datetime object to the nearest 3-hour block.
    rounded_hour = dt.hour - (dt.hour % 3)
    return dt.replace(hour=rounded_hour, minute=0, second=0, microsecond=0)


class CreateBatchProcessor:
    """
    A class to process image metadata for batch processing.

    Attributes:
    - ykeys (dict): Dictionary of keys from the YAML configuration.
    - csv_path (Path): Path to the CSV file containing image metadata.
    - df (pd.DataFrame): DataFrame holding the processed image metadata.

    Methods:
    - read_and_convert_datetime(): Reads CSV and converts specified columns to datetime.
    - split_datetime(): Splits the datetime into separate date and time columns.
    - preprocess_df(): Filters and cleans the DataFrame.
    - adjust_groups(): Adjusts DataFrame for batch processing.
    - config_keys(): Configures keys from the YAML file.
    - move_from_weeedsimagerepo2fieldbatches(batch): Moves batches from one repo to another.
    - process_df(): Processes DataFrame without concurrency.
    - process_df_concurrently(): Processes DataFrame with concurrency.
    """

    def __init__(self, cfg: DictConfig) -> None:
        log.info("Initializing CreateBatchProcessor")
        self.ykeys = read_yaml(cfg.pipeline_keys)
        self.datadir = Path(cfg.data.datadir, "processed_tables")
        self.csv_path = Path(
            find_most_recent_csv(self.datadir, "merged_blobs_tables_metadata.csv")
        )
        
        self.permanent_csv = Path(cfg.data.permanent_datadir,"merged_blobs_tables_metadata_permanent.csv")
        
        self.file_path = "./tempoutputfieldbatches.txt"
        log.info(f"CSV Path: {self.csv_path}")
        self.read_and_convert_datetime()
    
    def replace_date_format(self, date):
        # Define the regex pattern to match the date format
        pattern = r'(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})'
        # Replace ':' with '-' using regex substitution
        if pd.isna(date):
            return date
        return re.sub(pattern, r'\1-\2-\3 \4:\5:\6', date)
    
    def standardize_dt_format(self, df):
        df['CameraInfo_DateTime'] = df['CameraInfo_DateTime'].apply(self.replace_date_format)
        df['CameraInfo_DateTime'] = pd.to_datetime(df['CameraInfo_DateTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        return df

    def read_and_convert_datetime(self) -> pd.DataFrame:
        log.info("Reading and converting datetime columns in CSV")
        log.info(f"Reading path: {self.csv_path}")
        # self.df = read_csv_as_df(self.csv_path)
        self.df = pd.read_csv(self.csv_path, dtype={'CameraInfo_DateTime': str})
        if "CameraInfo_DateTime" not in self.df.columns:
            # Reads a CSV file into a DataFrame and converts specified columns to datetime.
            self.df = pd.read_csv(self.permanent_csv, dtype={'CameraInfo_DateTime': str})

        self.df = self.standardize_dt_format(self.df)

    def split_datetime(self) -> "CreateBatchProcessor":
        log.info("Splitting datetime into separate date and time columns")
        self.df["CameraInfo_Date"] = self.df["CameraInfo_DateTime"].dt.date
        self.df["CameraInfo_Time"] = self.df["CameraInfo_DateTime"].dt.time
        return self

    def preprocess_df(self) -> "CreateBatchProcessor":
        # Filters and cleans the DataFrame based on matching JPG and RAWs, and JPG extension.
        log.info("Preprocessing DataFrame")
        self.df = self.df[self.df["HasMatchingJpgAndRaw"] == True].dropna(
            subset=["UsState"]
        )
        self.df = self.df[self.df["Extension"] == "jpg"]
        self.df = self.df.dropna(subset="CameraInfo_DateTime")
        return self

    def adjust_groups(self) -> pd.DataFrame:
        # Adjusts DataFrame to group data for batch processing.
        # fmt: off
        # Sort the DataFrame to ensure correct comparison and grouping
        log.info("Adjusting groups for batch processing")
        self.df = self.df.sort_values(by=['UsState', 'MasterRefID', 'CameraInfo_DateTime'])
        
        # Apply this rounding to the CameraInfo_DateTime column to create a 3-hourly grouping column
        self.df['ThreeHourlyGroup'] = self.df['CameraInfo_DateTime'].apply(round_down_to_nearest_3_hours)
        self.df['SubBatchIndex'] = self.df.sort_values(by=['ThreeHourlyGroup']).groupby(['UsState', 'CameraInfo_Date'])['ThreeHourlyGroup'].transform(lambda x: pd.factorize(x)[0] + 1)
        self.df['SubBatchIndex_Padded'] = self.df['SubBatchIndex'].apply(lambda x: f"{x:0{2}d}")
        self.df =  self.df.sort_values(by=["UsState", "CameraInfo_Date", "SubBatchIndex"])
        self.df['batches'] = self.df.apply(lambda row: f"{row['UsState']}_{row['CameraInfo_Date'].strftime('%Y-%m-%d')}/raws/{row['SubBatchIndex_Padded']}/{row['Name'].replace('JPG', 'ARW')}", axis=1)
        return self

    def config_keys(self) -> None:
        # Configures and stores the necessary keys for batch processing.
        # fmt: off
        log.info("Configuring keys for batch processing")
        self.write_fbatch_key = self.ykeys["blobs"]["field-batches"]["write_sas_token"]
        self.fbatch_url = self.ykeys["blobs"]["field-batches"]["url"]

        self.read_weedimgrepo_key = self.ykeys["blobs"]["weedsimagerepo"]["sas_token"]
        self.weedimgrepo_url = self.ykeys["blobs"]["weedsimagerepo"]["url"]

    def add_extra_number(self, cell_value):
        # Extract the last number from the cell
        last_number = int(cell_value[-1])
        # Increment the last number by 1 and pad with zeros
        new_last_number = str(last_number + 1).zfill(len(cell_value))
        # Concatenate the original value with the new last number
        return cell_value[:-1] + new_last_number
    
    def filter_batched_data(self,present_batches_df):
        self.df = self.df[~self.df["BaseName"].isin(present_batches_df["BaseName"])]
        if len(self.df) == 0:
            log.info("No new images present. No images to be moved to the field-batches blob containers. Exiting.")
            exit(0)

        self.df[["BatchID_y", "Subfolder1", "Subfolder2", "FName"]] = self.df['batches'].str.split('/', expand=True)
        self.df["BatchFolder"] = self.df["BatchID_y"] + "/" + self.df["Subfolder1"] + "/" + self.df["Subfolder2"]
        
        duplicate_batch_folders = self.df[self.df["BatchFolder"].isin(present_batches_df["BatchFolder"])]
        
        if len(duplicate_batch_folders) > 0:
            
            log.warning("Duplicate batch folders. Renaming the duplicates. Duplicates include...")
            for index, row in duplicate_batch_folders.iterrows():
                log.warning(f'{row["BatchFolder"]}/{row["FName"]}')
            log.warning("Consider investiagting duplicate folders. Exiting.")
            exit()
            



    def move_from_weeedsimagerepo2fieldbatches(self, batch: str) -> None:
        # Moves batches from the weeds image repository to field batches using azcopy.
        log.info(f"Moving batch {batch} from weeds image repository to field batches")
        blob_name = Path(batch).name
        weedimgrepo_src = (
            f"{self.weedimgrepo_url}/{blob_name}{self.read_weedimgrepo_key}"
        )

        new_blob_name = batch
        fieldbatch_dst = f"{self.fbatch_url}/{new_blob_name}{self.write_fbatch_key}"

        command = f'azcopy copy "{weedimgrepo_src}" "{fieldbatch_dst}" --recursive'
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            log.info("Copy successful")
            log.debug(result.stdout)
        else:
            log.error("Error in copy operation")
            log.debug(result.stderr)

    def process_df(self) -> None:
        log.info("Processing DataFrame without concurrency")
        for idx, row in self.df.reset_index().iterrows():
            batch = row["batches"]
            self.move_from_weeedsimagerepo2fieldbatches(batch)

    def process_df_concurrently(self) -> None:
        # Processes the DataFrame using concurrency to handle multiple batches simultaneously.
        log.info("Processing DataFrame with concurrency")
        batches = self.df["batches"].unique()
        max_workers = int(len(os.sched_getaffinity(0)) / 3)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Schedule the batch processing tasks and execute them concurrently
            futures = [
                executor.submit(self.move_from_weeedsimagerepo2fieldbatches, batch)
                for batch in batches
            ]

            # Optionally, if you want to handle the results or exceptions
            for future in as_completed(futures):
                try:
                    future.result()  # If the function returns something you can capture it here
                except Exception as e:
                    log.error(f"Error processing batch: {e}")


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    present_batches_df = FieldBatchLister(cfg).df
    present_batches_df.to_csv("present_batches.csv", index=False)
    dataproc = CreateBatchProcessor(cfg)
    dataproc.config_keys()
    dataproc.split_datetime()
    dataproc.preprocess_df()
    dataproc.adjust_groups()
    dataproc.filter_batched_data(present_batches_df)
    
    # dataproc.df.to_csv("adjusted_groups.csv", index=False)
    run_concurrent = True
    if run_concurrent:
        dataproc.process_df_concurrently()
    else:
        dataproc.process_df()

    log.info(f"{cfg.general.task} completed.")
