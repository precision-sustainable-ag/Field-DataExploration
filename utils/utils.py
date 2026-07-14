import re
import subprocess
from datetime import datetime
from pathlib import Path
import requests

import exifread
import pandas as pd
import yaml


def read_yaml(path: str) -> dict:
    """Reads a YAML file and returns its content as a dictionary."""
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")


def read_csv_as_df(path: str) -> pd.DataFrame:
    """Reads a CSV file into a pandas DataFrame."""
    try:
        csv_reader = pd.read_csv(path, low_memory=False)
        # Return as dataframe
        return csv_reader
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")



def find_most_recent_data_csv(root_dir, filename="merged_blobs_tables_metadata.csv"):
    # Regular expression to match folder names and extract dates
    folder_name_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    dirs = [x for x in Path(root_dir).rglob(f"*/{filename}")]
    most_recent_date = None
    target_folder = None
    for i in dirs:
        for j in i.parts:
            if folder_name_pattern.match(str(j)):
                folder_date = datetime.strptime(j, "%Y-%m-%d")
                if most_recent_date is None or folder_date > most_recent_date:
                    most_recent_date = folder_date
                    target_folder = Path(i)

    if target_folder:
        return target_folder
    return None


def get_exif_data(image_path: str) -> dict:
    """Extracts EXIF data from an image file and returns it as a dictionary."""
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f)
        if tags:
            exif = {}
            for k, v in tags.items():
                if k not in (
                    "JPEGThumbnail",
                    "TIFFThumbnail",
                    "Filename",
                    "EXIF MakerNote",
                ):
                    if isinstance(v, (int, float)):
                        # Integers and floats are left as is
                        value = v
                    else:
                        # Convert other types to string as a general case
                        value = str(v)
                    if "Thumbnail" in k:
                        continue
                    exif[k] = value
        else:
            exif = {}
    return exif



def azcopy_list(url, read_keys, tempoutput):
    azlist_src = url + read_keys
    command = f'azcopy list "{azlist_src}" > {tempoutput}'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Check if the command was executed successfully
    if result.returncode != 0:
        print("Copy unsuccessful")
        print(result.stdout)


def download_azcopy(azuresrc, localdest):
    command = f'azcopy cp "{azuresrc}" "{localdest}"'

    # result = subprocess.run(command, capture_output=True, text=True)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Check if the command was executed successfully
    if result.returncode == 0:
        print("Copy successful")
        print(result.stdout)
    else:
        print("Error in copy operation")
        print(result.stderr)

def download_azcopy_multiple(azuresrc, localdest, image_list):
    command = f'azcopy cp "{azuresrc}" "{localdest}" --include-path="{image_list}"'

    # result = subprocess.run(command, capture_output=True, text=True)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Check if the command was executed successfully
    if result.returncode == 0:
        print("Copy successful")
        print(result.stdout)
    else:
        print("Error in copy operation")
        print(result.stderr)
        
def download_from_url(image_url: str, savedir: str = ".") -> None:
    """Downloads an image from a URL and saves it to the specified directory."""
    if not Path(savedir).exists():
        Path(savedir).mkdir(exist_ok=True, parents=True)
    fname = Path(image_url).name
    fpath = Path(savedir, fname)
    # Send a GET request to the image URL
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary write mode
        with open(fpath, "wb") as file:
            # Write the content of the response to the file
            file.write(response.content)
    else:
        print(f"Failed to download image from {image_url}")