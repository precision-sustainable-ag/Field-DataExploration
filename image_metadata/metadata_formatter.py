import json
import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from utils.data_anonymizer import DataAnonymizer
from utils.utils import find_most_recent_csv, get_exif_data

log = logging.getLogger(__name__)


class MetadataFormatter:
    """
    A class designed to format field metadata, including operations such as cleaning
    DataFrame rows, excluding specific keys, categorizing dictionary data, encrypting
    names, and inserting EXIF data from images.

    Attributes:
        keys_path (str): Path to the encryption keys.
        main_directory_path (Path): Parent directory path of the processed data directory.
        image_dir (str): Directory containing JPEG images.
        temp_json_dir (Path): Temporary directory for storing JSON files.
        csv_filename (str): Filename of the CSV containing merged blobs tables metadata.
        df (DataFrame): Pandas DataFrame containing the most recent table data.

    Methods:
        _get_most_recent_table: Retrieves the most recent CSV data file.
        cleaning_df: Cleans the DataFrame by replacing values and removing unnamed columns.
        filter_df_for_jpgs_only: Filters the DataFrame for JPEG files only.
        clean_df_row: Cleans individual DataFrame rows and encrypts the 'Username' field.
        exclude_keys: Excludes specified keys from a dictionary.
        categorize_dict: Categorizes dictionary data based on key prefixes.
        encrypt_name: Encrypts the 'Username' field in a row.
        insert_exif_data: Inserts EXIF data into a row from an image.
        write_json: Writes a dictionary to a JSON file.
        convert2dicts: Converts DataFrame rows to dictionaries and processes them.
        get_data: Filters DataFrame for JPEGs, processes, and converts rows to dictionaries.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the FieldMetadataFormatter with configuration settings.

        Parameters:
            cfg (DictConfig): Configuration settings.
        """
        self.main_directory_path = Path(cfg.data.processed_datadir).parent
        self.image_dir = cfg.temp.temp_jpegs_data
        self.temp_json_dir = Path(cfg.temp.temp_json_data)
        self.temp_json_dir.mkdir(exist_ok=True, parents=True)
        self.csv_filename = "merged_blobs_tables_metadata.csv"
        self.df = self._get_most_recent_table()
        self.anonymizer = DataAnonymizer(cfg.pipeline_keys)

    def _get_most_recent_table(self) -> pd.DataFrame:
        """Finds and loads the most recent CSV data file based on naming conventions and directory structure."""
        most_recent_csv_path = find_most_recent_csv(
            self.main_directory_path, self.csv_filename
        )
        return pd.read_csv(most_recent_csv_path)

    def cleaning_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans the DataFrame."""
        # Replace numbers with strings in the 'SizeClass' column
        df["SizeClass"] = df["SizeClass"].replace(
            {"3": "LARGE", "2": "MEDIUM", "1": "SMALL"}
        )
        # Remove unnamed columns
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        # Remove unnamed columns
        if "PartitionKey" in df.columns:
            df = df.drop(columns=["PartitionKey"])

        return df

    def filter_df_for_jpgs_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters the DataFrame for JPEG files only."""
        # Extract base name and extension
        df["extension"] = df["name"].str.split(".", n=1).str[-1]
        df["extension"] = df["extension"].str.lower()
        df = df[(df["extension"] == "jpg")].drop_duplicates(subset="name")
        df = self.cleaning_df(df)
        return df

    def clean_df_row(self, row: pd.Series) -> pd.Series:
        """Cleans individual DataFrame rows."""
        if type(row["Height"]) is str:
            row["Height"] = row["Height"].replace("\u2013", "-")
        if type(row["GroundCover"]) is str:
            row["GroundCover"] = row["GroundCover"].replace("\u2013", "-")

        row = self.encrypt_name(row)
        return row

    def exclude_keys(self, data: dict) -> dict:
        """Excludes specified keys from a dictionary."""
        excluded_keys = [
            "Image XResolution",
            "Image YResolution",
            "Image ResolutionUnit",
            "Image GPSInfo",
            "EXIF RecommendedExposureIndex",
            "EXIF DateTimeDigitized",
            "EXIF OffsetTime",
            "EXIF OffsetTimeOriginal",
            "EXIF OffsetTimeDigitized",
            "EXIF UserComment",
            "EXIF ExifImageWidth",
            "EXIF ExifImageLength",
            "Interoperability InteroperabilityIndex",
            "Interoperability InteroperabilityVersion",
            "EXIF InteroperabilityOffset",
            "EXIF FileSource",
            "EXIF SceneType",
            "EXIF CustomRendered",
            "EXIF SceneCaptureType",
            "GPS GPSVersionID",
            "GPS GPSLatitudeRef",
            "GPS GPSLatitude",
            "GPS GPSLongitudeRef",
            "GPS GPSLongitude",
            "GPS GPSTimeStamp",
            "GPS GPSStatus",
            "GPS GPSMeasureMode",
            "GPS GPSMapDatum",
            "GPS GPSDate",
            "GPS GPSDifferential",
        ]
        # List of keys you want to exclude
        filtered_dict = {k: v for k, v in data.items() if k not in excluded_keys}
        return filtered_dict

    def categorize_dict(self, data: dict) -> dict:
        """Categorizes dictionary data based on key prefixes."""
        for old_key in list(data.keys()):
            # Assign key-value pairs to the appropriate category
            if old_key.startswith("EXIF"):
                new_key = old_key.replace("EXIF ", "")
                data[new_key] = data.pop(old_key)

            elif old_key.startswith("Image"):
                new_key = old_key.replace("Image ", "")
                data[new_key] = data.pop(old_key)

        return data

    def encrypt_name(self, row: pd.Series) -> pd.Series:
        """Encrypts the 'Username' field in a row."""
        name = row["Username"]
        if not pd.isna(name):
            row["Username"] = self.anonymizer.encrypt(name)
        return row

    def insert_exif_data(self, row: pd.Series) -> pd.Series:
        """Inserts EXIF data into a row from an image."""
        name = row["name"]
        imgpath = Path(self.image_dir, name)
        exif_data = get_exif_data(imgpath)
        exif_data = self.exclude_keys(exif_data)
        exif_data = self.categorize_dict(exif_data)

        for key, value in exif_data.items():
            row[key] = value
        return row

    def write_json(self, row_dict: dict) -> None:
        """Writes a dictionary to a JSON file."""
        # Save the dictionary to a JSON file
        json_path = Path(self.temp_json_dir, Path(row_dict["name"]).stem + ".json")
        with open(json_path, "w") as f:
            json.dump(row_dict, f, indent=4, allow_nan=False)

    def convert2dicts(self, df: pd.DataFrame) -> None:
        """Converts DataFrame rows to dictionaries and processes them."""
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # Convert row to dictionary
            row = self.clean_df_row(row)
            row = self.insert_exif_data(row)
            row_dict = {k: v if pd.notna(v) else None for k, v in row.to_dict().items()}
            # You might want to use a more unique identifier depending on your DataFrame
            self.write_json(row_dict)

    def get_data(self) -> None:
        """Filters DataFrame for JPEGs, processes, and converts rows to dictionaries."""
        df = self.filter_df_for_jpgs_only(self.df)
        row_dicts = self.convert2dicts(df)


def main(cfg: DictConfig) -> None:
    """
    Main function to execute the FieldMetadataFormatter class functionalities.

    Parameters:
        cfg (DictConfig): Configuration settings.
    """
    log.info(f"Starting {cfg.general.task}")

    tableformatter = MetadataFormatter(cfg)
    tableformatter.get_data()
