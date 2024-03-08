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


class UnnestedDict:
    @staticmethod
    def unnest(
        d: Dict[str, Any], parent_key: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        """Recursively unnests a dictionary, prefixing parent keys to child keys."""

        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(UnnestedDict.unnest(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class ImageMetadataProcessor:

    def __init__(self, cfg):
        """Initializes the ImageMetadataProcessor with paths and loads the dataframe."""
        self.csvpath = find_most_recent_csv(
            Path(cfg.data.processed_datadir).parent,
            "merged_blobs_tables_metadata.csv",
        )

        self.imgdir = Path(cfg.temp.temp_jpegs_data)
        self.jsondir = Path(cfg.temp.temp_json_data)
        self.jsondir.mkdir(exist_ok=True, parents=True)
        self.df = None
        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        """Loads and prepares the image metadata dataframe."""
        self.df = read_csv_as_df(self.csvpath)
        self.df = self.df.replace(np.nan, None)
        self.df = self.df.drop(["Username", "ImageURL"], axis=1)

    def process_images(self):
        """Processes each image, extracting and saving its metadata as JSON."""
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            self.clean_row_data(row)
            jpeg_path = self.imgdir / row["Name"]
            if not jpeg_path.exists():
                continue
            exif_data = get_exif_data(jpeg_path)
            caminfo = CameraInfo.from_dict(exif_data)
            row_dict = row.to_dict()
            row_dict["CameraInfo"] = caminfo
            fmetadata = FieldMetadata(**row_dict)
            self.save_json_metadata(fmetadata, row["Name"])

    def clean_row_data(self, row):
        """Cleans specific fields in a dataframe row in-place."""
        if isinstance(row["Height"], str):
            row["Height"] = row["Height"].replace("\u2013", "-")
        if isinstance(row["GroundCover"], str):
            row["GroundCover"] = row["GroundCover"].replace("\u2013", "-")

    def save_json_metadata(self, fmetadata, name):
        """Saves the field metadata as JSON."""
        json_path = self.jsondir / (Path(name).stem + ".json")
        with open(json_path, "w") as f:
            json.dump(asdict(fmetadata), f, indent=4, allow_nan=False, default=str)


class MetadataCSVConverter:
    """Initializes the MetadataCSVConverter with paths."""

    def __init__(self, cfg):
        self.jsondir = Path(cfg.temp.temp_json_data)
        self.output_csv = Path(cfg.data.processed_datadir, "public_metadata.csv")

    def convert_json_to_csv(self):
        """Converts JSON metadata files into a single CSV file."""
        rows = []
        for path in self.jsondir.glob("*"):
            with open(path, "r") as f:
                data = json.load(f)
                data = UnnestedDict.unnest(data)
                rows.append(pd.Series(data))
        fdf = pd.DataFrame(rows)
        fdf.to_csv(self.output_csv, index=False)


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    processor = ImageMetadataProcessor(cfg)
    processor.process_images()

    converter = MetadataCSVConverter(cfg)
    converter.convert_json_to_csv()
    log.info(f"{cfg.general.task} completed.")
