import logging
import os
import subprocess
import time
from functools import wraps
from pathlib import Path
import requests

import exifread
import pandas as pd
import yaml

log = logging.getLogger(__name__)


def retry_on_permission_error(max_attempts: int = 10, delay: float = 2.0):
    """Retry the wrapped call when it raises PermissionError.

    The /mnt NFS share (Kerberos-secured autofs) sometimes returns a
    transient "Permission denied" on the very first touch of a given path
    per session, while the mount/ticket is still settling - happens to every
    user, and a plain retry succeeds. An interactive shell "fixes" it by
    re-running the command by hand; cron doesn't get that chance, so it
    needs to retry itself.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except PermissionError:
                    if attempt == max_attempts:
                        raise
                    log.warning(
                        "Permission denied calling %s (attempt %d/%d) - likely the transient "
                        "/mnt NFS mount race, retrying in %.0fs",
                        func.__name__, attempt, max_attempts, delay,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


def warmup_mount_paths(paths, max_attempts: int = 10, delay: float = 2.0) -> None:
    """Touch each NFS path once, retrying through the transient permission-denied
    race so downstream pipeline tasks don't each have to handle it themselves."""
    @retry_on_permission_error(max_attempts=max_attempts, delay=delay)
    def _touch(path: str) -> None:
        os.listdir(path)

    seen = set()
    for path in paths:
        path = str(path)
        if path and path not in seen:
            seen.add(path)
            _touch(path)


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