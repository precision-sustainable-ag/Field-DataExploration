import logging
import os
import sys
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

log = logging.getLogger(__name__)

from utils.utils import (
    download_azcopy,
    download_from_url,
    read_csv_as_df,
    read_yaml,
    upload_azcopy,
)

path = "notebooks/data.csv"
yaml = "keys/authorized_keys.yaml"
outputdir = "tempdata/raw_data"

df = (
    read_csv_as_df(path)
    .sort_values(by=["CameraInfo_Date", "SubBatchIndex"])
    .reset_index()
)
yamlkeys = read_yaml(yaml)
download_yamlkeys = yamlkeys["blobs"]["weedsimagerepo"]["sas_token"]
upload_yamlkeys = yamlkeys["blobs"]["field-batches"]["write_sas_token"]
url_yamlkeys = yamlkeys["blobs"]["field-batches"]["url"]

for i, row in tqdm(df.iterrows(), total=len(df)):
    state = row["UsState"]
    date = row["CameraInfo_Date"]
    name = row["Name"].replace("JPG", "ARW")
    group = row["SubBatchIndex"]

    localdest = Path(outputdir, f"{state}_{date}_{group:02d}")
    localdest.mkdir(exist_ok=True, parents=True)
    imgurlraw = row["ImageURL"].replace("JPG", "ARW")
    imgurljpg = row["ImageURL"]
    azuresrcraw = f"{imgurlraw}{download_yamlkeys}"
    azuresrcjpg = f"{imgurljpg}{download_yamlkeys}"
    download_azcopy(azuresrcraw, localdest)
    download_azcopy(azuresrcjpg, localdest)

    # azcopy cp "/path/to/file.txt" "https://[account].blob.core.windows.net/[container]/[path/to/blob]?[SAS]"
    localdestpath = Path(outputdir, f"{state}_{date}_{group:02d}", name)  # path to file

    azure_path = Path(f"{state}_{date}/raws/{group:02d}", name)
    azuredst = f"{url_yamlkeys}/{azure_path}{upload_yamlkeys}"

    upload_azcopy(localdestpath, azuredst)

    # os.remove(localdestpath)
    with open("uploaded_images.txt", "a") as f:
        f.write(f"{localdestpath}\n")

    if i == 10:
        break
