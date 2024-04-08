import concurrent.futures
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
from omegaconf import DictConfig

from utils.utils import (
    azcopy_list,
    download_azcopy,
    find_most_recent_csv,
    get_exif_data,
    read_csv_as_df,
    read_yaml,
)

log = logging.getLogger(__name__)


class ReportWriter:
    """
    Provides functionality for writing a list of stems (unique identifiers for images) to a file.

    Methods:
    --------
    write_stems(stems, output_file_path: str) -> None:
        Writes the provided list of stems to the specified file path. Each stem is written on a new line.

    This class is intended to be used where a simple utility is needed to output stems to a file, often for logging or tracking purposes.
    """

    @staticmethod
    def write_stems(stems, output_file_path: str) -> None:
        log.info("Writing stems to %s.", output_file_path)
        try:
            with open(output_file_path, "w") as output_file:
                for stem in stems:
                    output_file.write(f"{stem}\n")
            log.info("Stems successfully written to %s.", output_file_path)
        except Exception as e:
            log.error(
                "Failed to write stems to %s: %s", output_file_path, e, exc_info=True
            )
            raise


class DataProcessor:
    """
    Base class for data processing, providing a structure for subclasses that implement specific data processing tasks.

    read_and_preprocess(csv_path: Path) -> pd.DataFrame:
        Abstract method for reading and preprocessing data from a CSV file. Must be implemented by subclasses.

    download_jpgs(imgurl: str):
        Abstract method for downloading JPG images. Must be implemented by subclasses.

    The DataProcessor class serves as a foundation for building specific data processing functionalities that share common configuration handling but require different processing implementations.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def read_and_preprocess(self, csv_path: Path) -> pd.DataFrame:
        raise NotImplementedError

    def download_jpgs(self, imgurl: str):
        raise NotImplementedError


class WeedsImageRepoDataProcessor(DataProcessor):
    """
    Processes data specific to the Weeds Image Repository, including reading and preprocessing CSV data, extracting stem lists, and downloading JPG images.

    Attributes:
    -----------
    Inherits all attributes from the DataProcessor class and introduces additional ones for handling Weeds Image Repository specific data processing tasks.

    Methods:
    --------
    Inherits methods from DataProcessor and overrides `read_and_preprocess` and `download_jpgs` with specific implementations. It also adds methods for extracting EXIF data, updating dataframes with this data, and saving the updated dataframe.

    """

    # fmt: off
    def __init__(self, cfg: DictConfig):
        
        super().__init__(cfg)
        
        self.datadir = Path(cfg.data.datadir, "processed_tables")
        self.csv_path = Path(find_most_recent_csv(self.datadir, "merged_blobs_tables_metadata.csv"))
        self.updated_csv_path = self.csv_path
        self.keys = cfg.pipeline_keys
        self.df = self.read_and_preprocess(self.csv_path)
        self.config_keys()
        self.local_destination = Path("tempdata/jpg_data")
        self.local_destination.mkdir(exist_ok=True, parents=True)

        log.info("WeedsImageRepoDataProcessor initialized successfully.")
            

    def config_keys(self):
        # Configures keys from YAML configuration.
        yamkeys = read_yaml(self.keys)
        self.wir_sas_token = yamkeys["blobs"]["weedsimagerepo"]["sas_token"]
        self.wir_url = yamkeys["blobs"]["weedsimagerepo"]["url"]
        log.debug("WeedsImageRepo keys configured.")

    def read_and_preprocess(self, csv_path: Path) -> pd.DataFrame:
        # Reads and preprocesses the CSV file at csv_path.
        log.info("Reading and preprocessing CSV data from %s.", csv_path)
        df = read_csv_as_df(csv_path)
        df["Stem"] = df["Name"].str.extract(r"(.*)(?=\.)")
        log.debug("CSV data read and preprocessed successfully.")
        return df

    def get_stem_list(self) -> Set[str]:
        # Returns a set of stems with matching JPG and Raw files and with non-null US State.
        stem_list = set(self.df.loc[self.df["HasMatchingJpgAndRaw"] & self.df["UsState"].notna(), "Stem"])
        log.info("Extracted stem list with %d entries.", len(stem_list))
        return stem_list

    def get_missing_jpg(self, stem_list: Set[str]) -> pd.DataFrame:
        # Returns a DataFrame of JPGs missing from the specified stem list.
        missing_jpg_df = self.df.loc[self.df["Stem"].isin(stem_list) & (self.df["Extension"] == "jpg")]
        self.df.drop(columns=["Stem"], inplace=True)
        log.info("%d missing JPGs identified.", len(missing_jpg_df))
        return missing_jpg_df

    def download_jpgs(self, imgurl: str, destination_path: str) -> None:
        azuresrc = imgurl + self.wir_sas_token
        # Assuming download_azcopy can accept a destination file path directly
        # If not, you may need to modify it to support downloading to a specific path
        log.debug("Downloading JPG for %s.", imgurl)
        download_azcopy(azuresrc, destination_path)
    
    def extract_exifdatetime(self, row: pd.Series) -> str:
        # Extracts the EXIF DateTimeOriginal from the JPG specified in the row.
        # Utilizes a temporary file to ensure automatic cleanup.
        # Use NamedTemporaryFile within a context manager to ensure cleanup
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            temp_jpeg_path = tmp_file.name
            # Download the image directly to the temporary file
            self.download_jpgs(row["ImageURL"], temp_jpeg_path)

        try:
            # Now that the file is closed, you can read the EXIF data
            exif_datetime = get_exif_data(temp_jpeg_path)["EXIF DateTimeOriginal"]
            log.debug("EXIF DateTimeOriginal extracted for %s.", row["Name"])
        finally:
            # Ensure the temporary file is removed after processing
            os.remove(temp_jpeg_path)

        return exif_datetime
    

    def update_dataframe_with_exif_data(self, missing_jpgs_df: pd.DataFrame) -> None:
        log.info("Updating DataFrame with EXIF DateTimeOriginal data.")

        def update_row(row):
            try:
                exif_datetime = self.extract_exifdatetime(row)
                return row["ImageURL"], exif_datetime
            except Exception as e:
                log.error("Failed to update EXIF data for image %s: %s", row["ImageURL"], e, exc_info=True)
                return row["ImageURL"], None
        max_workers = int(len(os.sched_getaffinity(0)) / 3)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(update_row, row) for _, row in missing_jpgs_df.iterrows()]
            for future in concurrent.futures.as_completed(futures):
                img_url, exif_datetime = future.result()
                if exif_datetime:
                    self.df.loc[self.df["ImageURL"] == img_url, "CameraInfo_DateTime"] = exif_datetime
                    log.info("EXIF data updated for image: %s", img_url)


    def save_updated_dataframe(self) -> None:
        # Saves the updated DataFrame to a CSV file.
        try:
            self.df.to_csv(self.updated_csv_path, index=False)
            log.info("DataFrame saved successfully to %s.", self.updated_csv_path)
        except Exception as e:
            log.error("Failed to save DataFrame to %s: %s", self.updated_csv_path, e, exc_info=True)

    def merge_batch_info(self, df):
        # self.df['SubBatchIndex'] = self.df['SubBatchIndex'].astype(str) if 'SubBatchIndex' in self.df.columns else self.df['SubBatchIndex']
        self.df.drop(
            [
                col
                for col in self.df.columns
                if ("SubBatchIndex" in col) or ("BatchID" in col)
            ],
            axis=1,
            inplace=True,
        )
        df['SubBatchIndex'] = df['SubBatchIndex'].astype(str)
        self.df = pd.merge(self.df, df, on="Name", how="left")
        self.df['SubBatchIndex'] = self.df['SubBatchIndex'].astype(str)


class FieldBatchesDataProcessor(DataProcessor):
    """
    Processes field batch data by listing blob (field-batches) contents and extracting raw stems from the data.

    Methods:
    --------
    list_blob_contents() -> None:
        Lists the contents of a blob storage, logging the output to a specified file path.

    get_raw_stems(file_path: Path) -> Set[str]:
        Extracts and returns a set of raw stems from the listed blob contents.

    The FieldBatchesDataProcessor class is focused on processing data related to field batches, specifically in the context of managing and analyzing data stored in blob storage. It provides tools for listing storage contents and extracting relevant information for further processing.
    """

    # Processes field batch data.
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.yamlkeys = read_yaml(cfg.pipeline_keys)
        self.file_path = "./tempoutputfieldbatches.txt"
        log.info("FieldBatchesDataProcessor initialized successfully.")

    def list_blob_contents(self) -> None:
        # Lists the contents of the blob storage
        log.info("Listing blob contents.")
        try:
            read_yamlkeys = self.yamlkeys["blobs"]["field-batches"]["read_sas_token"]
            url_yamlkeys = self.yamlkeys["blobs"]["field-batches"]["url"]
            azcopy_list(url_yamlkeys, read_yamlkeys, self.file_path)
            log.info("Blob contents successfully listed.")
        except Exception as e:
            log.error("Failed to list blob contents: %s", e, exc_info=True)
            raise

    def get_raw_stems(self, file_path: Path) -> Set[str]:
        # Extracts stems from the listed blob contents.
        stems = set()
        with open(self.file_path, "r") as file:
            for line in file:
                parts = line.split("/")
                if len(parts) > 1 and "raws" in parts:
                    stem = parts[-1].split(".")[0]
                    stems.add(stem)
        log.info("%d raw stems identified.", len(stems))
        return stems

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
        os.remove(self.file_path)
        return df


def main(cfg: DictConfig) -> None:
    # Main function to orchestrate the data processing.
    log.info(f"Starting {cfg.general.task}")
    try:
        weedrepoproc = WeedsImageRepoDataProcessor(cfg)
        wir_stems = weedrepoproc.get_stem_list()

        fieldbatchpro = FieldBatchesDataProcessor(cfg)
        fieldbatchpro.list_blob_contents()
        field_stems = fieldbatchpro.get_raw_stems(Path(fieldbatchpro.file_path))
        df = fieldbatchpro.current_batches_in_fieldbatches()

        missing_stems = wir_stems - field_stems
        miss_jpgs_df = weedrepoproc.get_missing_jpg(missing_stems)

        weedrepoproc.update_dataframe_with_exif_data(miss_jpgs_df)
        weedrepoproc.merge_batch_info(df)
        weedrepoproc.save_updated_dataframe()

        log.info(f"{cfg.general.task} completed.")
    except Exception as e:
        log.error(f"An error occurred during {cfg.general.task}: {e}", exc_info=True)
