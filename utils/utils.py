import os
from datetime import datetime
from pathlib import Path

import exifread
import pandas as pd
import requests
import yaml

from utils.metadata_dataclass import CameraInfo


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


def find_most_recent_csv(main_directory_path: str, csv_filename: str) -> str | None:
    """
    Finds the most recent CSV file based on the subfolder names which are dates.
    """
    # List all items in the main directory
    all_items = os.listdir(main_directory_path)

    # Filter out items that are not directories or don't match the date pattern
    dated_subfolders = [
        item
        for item in all_items
        if os.path.isdir(os.path.join(main_directory_path, item))
    ]
    # Ensure that the folder names are valid dates
    dated_subfolders = [
        folder
        for folder in dated_subfolders
        if len(folder) == 10 and folder.count("-") == 2
    ]

    # Sort the list of dated subfolders to find the most recent one
    dated_subfolders.sort(
        key=lambda date: datetime.strptime(date, "%Y-%m-%d"), reverse=True
    )

    # The first element is now the most recent subfolder
    most_recent_subfolder = dated_subfolders[0] if dated_subfolders else None

    if most_recent_subfolder:
        # Construct the path to the most recent CSV file
        most_recent_csv_path = os.path.join(
            main_directory_path, most_recent_subfolder, csv_filename
        )
        return most_recent_csv_path

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
