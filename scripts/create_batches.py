import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Set

import pandas as pd
from omegaconf import DictConfig

from utils.utils import azcopy_list, find_most_recent_csv, read_csv_as_df, read_yaml

log = logging.getLogger(__name__)


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
        self.file_path = "./tempoutputfieldbatches.txt"
        log.info(f"CSV Path: {self.csv_path}")
        self.read_and_convert_datetime()

    def update_merged_table_with_batchinfo(self):
        self.list_blob_contents()
        df = self.current_batches_in_fieldbatches()
        self.merge_batch_info(df)
        self.save_updated_dataframe()

    def list_blob_contents(self) -> None:
        # Lists the contents of the blob storage
        log.info("Listing blob contents.")
        try:
            read_yamlkeys = self.ykeys["blobs"]["field-batches"]["read_sas_token"]
            url_yamlkeys = self.ykeys["blobs"]["field-batches"]["url"]
            azcopy_list(url_yamlkeys, read_yamlkeys, self.file_path)
            log.info("Blob contents successfully listed.")
        except Exception as e:
            log.error("Failed to list blob contents: %s", e, exc_info=True)
            raise

    def current_batches_in_fieldbatches(self):
        content = []
        with open(self.file_path, "r") as file:
            for line in file:
                clean_line = line.replace("INFO: ", "")

                path = clean_line.split(";")[0]
                parts = path.split("/")
                if len(parts) > 1 and "raws" in parts:
                    content_dict = {}
                    content_dict["BatchID"] = parts[0]
                    content_dict["SubBatchIndex"] = str(parts[-2])
                    content_dict["Name"] = parts[-1]
                    content.append(content_dict)

        df = pd.DataFrame(content)
        df["SubBatchIndex"] = df["SubBatchIndex"].astype(str)
        return df

    def merge_batch_info(self, df):
        self.df.drop(
            [
                col
                for col in self.df.columns
                if ("SubBatchIndex" in col) or ("BatchID" in col)
            ],
            axis=1,
            inplace=True,
        )

        df["SubBatchIndex"] = df["SubBatchIndex"].astype(str)
        self.df = pd.merge(self.df, df, on="Name", how="left")
        os.remove(self.file_path)

    def save_updated_dataframe(self) -> None:
        # Saves the updated DataFrame to a CSV file.
        try:
            self.df.to_csv(self.csv_path, index=False)
            log.info("DataFrame saved successfully to %s.", self.csv_path)
        except Exception as e:
            log.error(
                "Failed to save DataFrame to %s: %s",
                self.csv_path,
                e,
                exc_info=True,
            )

    def read_and_convert_datetime(self) -> pd.DataFrame:
        log.info("Reading and converting datetime columns in CSV")
        log.info(f"Reading path: {self.csv_path}")
        self.df = read_csv_as_df(self.csv_path)
        self.df["UploadDateTimeUTC"] = pd.to_datetime(self.df["UploadDateTimeUTC"])
        if "CameraInfo_DateTime" in self.df.columns:
            # Reads a CSV file into a DataFrame and converts specified columns to datetime.
            self.df["CameraInfo_DateTime"] = pd.to_datetime(
                self.df["CameraInfo_DateTime"], format="%Y:%m:%d %H:%M:%S"
            )
        else:
            self.update_merged_table_with_batchinfo()
            log.error(
                f"The 'CameraInfo_DateTime' is not in merged_blobs_tables_metadata.csv. The 2 blob containers are probably up-to-date. Exiting."
            )
            exit(0)

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
    dataproc = CreateBatchProcessor(cfg)
    dataproc.config_keys()
    dataproc.split_datetime()
    dataproc.preprocess_df()
    dataproc.adjust_groups()
    run_concurrent = True
    if run_concurrent:
        dataproc.process_df_concurrently()
    else:
        dataproc.process_df()

    dataproc.update_merged_table_with_batchinfo()

    log.info(f"{cfg.general.task} completed.")
