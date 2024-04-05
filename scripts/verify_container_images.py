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
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def read_and_preprocess(self, csv_path: Path) -> pd.DataFrame:
        raise NotImplementedError

    def download_jpgs(self, imgurl: str):
        raise NotImplementedError


class WeedsImageRepoDataProcessor(DataProcessor):
    # fmt: off
    def __init__(self, cfg: DictConfig):
        
        super().__init__(cfg)
        
        self.datadir = Path(cfg.data.datadir, "processed_tables")
        self.check_for_updated_file()
        self.csv_path = Path(find_most_recent_csv(self.datadir, "merged_blobs_tables_metadata.csv"))
        self.updated_csv_path = Path(self.csv_path.parent, "merged_blobs_tables_metadata_updated_CameraDateTime.csv")
        self.keys = cfg.pipeline_keys
        self.df = self.read_and_preprocess(self.csv_path)
        self.config_keys()
        self.local_destination = Path("tempdata/jpg_data")
        self.local_destination.mkdir(exist_ok=True, parents=True)

        log.info("WeedsImageRepoDataProcessor initialized successfully.")

    def check_for_updated_file(self):
        path = find_most_recent_csv(self.datadir,"merged_blobs_tables_metadata_updated_CameraDateTime.csv")
        if Path(path).exists():
            log.critical("Exif data has alread been updated. Updated csv file already exists in the most up-to-date folder. Exiting.")
            exit(1)
            

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

    # def update_dataframe_with_exif_data(self, missing_jpgs_df: pd.DataFrame) -> None:
    #     # Updates the DataFrame with EXIF DateTimeOriginal data for the specified missing JPGs.
    #     log.info("Updating DataFrame with EXIF DateTimeOriginal data.")
    #     for _, row in missing_jpgs_df.iterrows():
    #         try:
    #             # self.download_jpgs(row)
    #             exif_datetime = self.extract_exifdatetime(row)
    #             self.df.loc[self.df["ImageURL"] == row["ImageURL"], "CameraInfo_DateTime"] = exif_datetime
    #             log.info("EXIF data updated for image: %s", row["ImageURL"])
    #         except Exception as e:
    #             log.error("Failed to update EXIF data for image %s: %s", row["ImageURL"], e, exc_info=True)


    def save_updated_dataframe(self) -> None:
        # Saves the updated DataFrame to a CSV file.
        try:
            self.df.to_csv(self.updated_csv_path, index=False)
            log.info("DataFrame saved successfully to %s.", self.updated_csv_path)
        except Exception as e:
            log.error("Failed to save DataFrame to %s: %s", self.updated_csv_path, e, exc_info=True)


class FieldBatchesDataProcessor(DataProcessor):
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
        with open(file_path, "r") as file:
            for line in file:
                parts = line.split("/")
                if len(parts) > 1 and "raws" in parts:
                    stem = parts[-1].split(".")[0]
                    stems.add(stem)
        log.info("%d raw stems identified.", len(stems))
        os.remove(self.file_path)
        return stems


def main(cfg: DictConfig) -> None:
    # Main function to orchestrate the data processing.
    log.info(f"Starting {cfg.general.task}")
    try:
        weedrepoproc = WeedsImageRepoDataProcessor(cfg)
        wir_stems = weedrepoproc.get_stem_list()

        fieldbatchpro = FieldBatchesDataProcessor(cfg)
        fieldbatchpro.list_blob_contents()
        field_stems = fieldbatchpro.get_raw_stems(Path(fieldbatchpro.file_path))

        missing_stems = wir_stems - field_stems
        miss_jpgs_df = weedrepoproc.get_missing_jpg(missing_stems)

        weedrepoproc.update_dataframe_with_exif_data(miss_jpgs_df)
        weedrepoproc.save_updated_dataframe()

        log.info(f"{cfg.general.task} completed.")
    except Exception as e:
        log.error(f"An error occurred during {cfg.general.task}: {e}", exc_info=True)
