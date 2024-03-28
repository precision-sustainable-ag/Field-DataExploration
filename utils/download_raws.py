import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
from utils.metadata_dataclass import CameraInfo, FieldMetadata
from utils.utils import find_most_recent_csv, get_exif_data, read_csv_as_df

log = logging.getLogger(__name__)
from utils.utils import download_from_url

path = "/home/mkutuga/Field-DataExploration/data/processed_tables/2024-03-19/public_metadata.csv"
path_urls = "/home/mkutuga/Field-DataExploration/data/processed_tables/2024-03-19/merged_blobs_tables_metadata.csv"
df = read_csv_as_df(path)
dfurls = read_csv_as_df(path_urls)

# Convert the column to datetime
df["UploadDateTimeUTC"] = pd.to_datetime(df["UploadDateTimeUTC"])
df["CameraInfo_DateTime"] = pd.to_datetime(
    df["CameraInfo_DateTime"], format="%Y:%m:%d %H:%M:%S"
)

# Split the datetime column into separate date and time columns
df["CameraInfo_Date"] = df["CameraInfo_DateTime"].dt.date
df["CameraInfo_Time"] = df["CameraInfo_DateTime"].dt.time
df = pd.merge(df, dfurls[["Name", "ImageURL"]], on="Name", how="left")

# # Only jpgs that have raws
# ds = df[df["HasMatchingJpgAndRaw"] == True].dropna(subset="UsState")

# # Filter for weeds only
# wds = ds[ds["PlantType"] == "WEEDS"].copy().sort_values(by="Name").reset_index()

# grouped_df = (
#     wds.groupby(
#         ["UsState", "CameraInfo_Date", "Species", "Name", "ImageURL", "MasterRefID"]
#     )
#     .size()
#     .reset_index(name="count")
#     .sort_values(by=["count"])
# )
# # group_sizes
# weeds_df = grouped_df.copy()
# # weeds_df = grouped_df[(grouped_df["count"]< 10)].sort_values(by="count", ascending=False)
# weeds_df["Group"] = (
#     weeds_df.groupby(["CameraInfo_Date", "Species"])["MasterRefID"]
#     .rank(method="dense")
#     .astype(int)
# )
# weeds_df["Group"] = weeds_df["Group"].apply("{:0>3}".format)
# weeds_df = weeds_df.sort_values(by="Name", ascending=False).reset_index(drop=True)

ds = df[df["HasMatchingJpgAndRaw"] == True].dropna(subset="UsState")
# Filter for weeds only
wds = ds[ds["PlantType"] == "WEEDS"].copy().sort_values(by="Name").reset_index()
grouped_df = (
    wds.groupby(["UsState", "CameraInfo_Date", "Species", "Name", "ImageURL"])
    .size()
    .reset_index(name="count")
    .sort_values(by=["count"])
)
weeds_df = grouped_df.sort_values(by="Name", ascending=False).reset_index(drop=True)

outputdir = Path(
    "/home/mkutuga/Field-DataExploration/tempdata/raw_images/weed_batches_groupedby_Species"
)
outputdir.mkdir(exist_ok=True, parents=True)

for idx, row in tqdm(weeds_df.iterrows(), total=len(weeds_df)):
    state = row["UsState"]
    date = row["CameraInfo_Date"]
    species = (
        row["Species"]
        if " " not in row["Species"]
        else row["Species"].replace(" ", "_")
    )
    name = row["Name"]
    batchdir = Path(outputdir, f"{state}_{date}", f"{species}")
    batchdir.mkdir(exist_ok=True, parents=True)
    image_url = row["ImageURL"].replace("JPG", "ARW")

    download_from_url(image_url, batchdir)
    if idx == 100:
        break


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")

    log.info(f"{cfg.general.task} completed.")
