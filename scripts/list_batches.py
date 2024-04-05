import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.utils import find_file_in_subdirectories, read_csv_as_df

# fmt: off
# Initialize logger
log = logging.getLogger(__name__)


def round_down_to_nearest_3_hours(dt: datetime) -> datetime:
    # Rounds down a datetime object to the nearest 3-hour block.
    rounded_hour = dt.hour - (dt.hour % 3)
    return dt.replace(hour=rounded_hour, minute=0, second=0, microsecond=0)


def map_to_sequential(group):
    unique_ids = group["AdjustedGroupID_3hr"].unique()
    id_map = {id_: idx + 1 for idx, id_ in enumerate(sorted(unique_ids))}
    group["SubBatchIndex"] = group["AdjustedGroupID_3hr"].map(id_map)
    return group


class DataProcessor:
    def __init__(self, path: str):
        self.df = self.read_and_convert_datetime(path)
        self.csv_save_path = "batches_in_weedsimagerepo.txt"

    def read_and_convert_datetime(self, path: str) -> pd.DataFrame:
        # Reads a CSV file into a DataFrame and converts specified columns to datetime.
        df = read_csv_as_df(path)
        # Convert the column to datetime
        df["UploadDateTimeUTC"] = pd.to_datetime(df["UploadDateTimeUTC"])
        df["CameraInfo_DateTime"] = pd.to_datetime(
            df["CameraInfo_DateTime"], format="%Y:%m:%d %H:%M:%S"
        )
        return df

    def split_datetime(self) -> "DataProcessor":
        self.df["CameraInfo_Date"] = self.df["CameraInfo_DateTime"].dt.date
        self.df["CameraInfo_Time"] = self.df["CameraInfo_DateTime"].dt.time
        return self

    def preprocess_df(self) -> 'DataProcessor':
        self.df = self.df[self.df["HasMatchingJpgAndRaw"] == True].dropna(subset=["UsState"])
        return self


    def adjust_groups(self) -> pd.DataFrame:
        # Sort the DataFrame to ensure correct comparison and grouping
        self.df = self.df.sort_values(by=['UsState', 'MasterRefID', 'CameraInfo_DateTime'])
        
        # Apply this rounding to the CameraInfo_DateTime column to create a 3-hourly grouping column
        self.df['ThreeHourlyGroup'] = self.df['CameraInfo_DateTime'].apply(round_down_to_nearest_3_hours)

        self.df['SubBatchIndex'] = self.df.sort_values(by=['ThreeHourlyGroup']).groupby(['UsState', 'CameraInfo_Date'])['ThreeHourlyGroup'].transform(lambda x: pd.factorize(x)[0] + 1)

        self.df['SubBatchIndex_Padded'] = self.df['SubBatchIndex'].apply(lambda x: f"{x:0{2}d}")


        self.df =  self.df.sort_values(by=["UsState", "CameraInfo_Date", "SubBatchIndex_Padded"])


        return self

    def write_batches(self, df):

        df[["UsState", "CameraInfo_Date", "SubBatchIndex_Padded"]].to_csv(self.csv_save_path.replace("txt", "csv"), index=False)
        
        df['batches'] = df.apply(lambda row: f"{row['UsState']}_{row['CameraInfo_Date'].strftime('%Y-%m-%d')}/raws/{row['SubBatchIndex_Padded']}/{row['Name'].replace('JPG', 'ARW')}", axis=1)
        with open(self.csv_save_path, 'w') as f:
            for i in df['batches']:
                f.write(f"{i}\n")



def main(cfg: DictConfig) -> None:
    log.info("Main process started with task: %s", cfg.general.task)
    
    csv_path = find_file_in_subdirectories(Path(cfg.data.datadir), "public_metadata.csv")
    dataproc = DataProcessor(csv_path)
    dataproc.preprocess_df()
    dataproc.split_datetime()
    dataproc.adjust_groups()
    df = dataproc.df
    dataproc.write_batches(df)
