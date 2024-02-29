import os
from datetime import datetime

import pandas as pd
import yaml


def read_yaml(path):
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")


def read_csv_as_df(path):
    try:
        csv_reader = pd.read_csv(path)
        # Return as dataframe
        return csv_reader
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")


def find_most_recent_csv(main_directory_path, csv_filename):
    """
    Finds the most recent CSV file based on the subfolder names which are dates.

    Parameters:
    - main_directory_path: str, the path to the main directory containing dated subfolders.
    - csv_filename: str, the filename of the CSV files contained in each subfolder.

    Returns:
    - The path to the most recent CSV file, or None if no valid subfolders or CSV file is found.
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
