import logging
import os
from typing import Any, Dict

from omegaconf import DictConfig

from utils.utils import azcopy_list, read_yaml

# Initialize logger
log = logging.getLogger(__name__)


class BlobStorageHandler:
    """
    Handles operations related to blob storage such as listing contents
    and analyzing folder structures.
    """

    def __init__(self, yaml: str, file_path: str):
        """
        Initializes the BlobStorageHandler with YAML configuration and file path.
        Logs the initialization.

        Args:
            yaml (str): Path to the YAML file containing storage access keys and URLs.
            file_path (str): Path to the file where blob content lists are stored.
        """
        self.yaml = yaml
        self.file_path = file_path
        log.info(
            "Initialized BlobStorageHandler with yaml: %s and file_path: %s",
            yaml,
            file_path,
        )

    def list_blob_contents(self) -> None:
        """
        Lists the contents of a blob storage based on configurations provided in a YAML file.
        Logs the start and end of the operation.
        """
        log.info("Listing blob contents.")
        try:
            yamlkeys = read_yaml(self.yaml)
            read_yamlkeys = yamlkeys["blobs"]["field-batches"]["read_sas_token"]
            url_yamlkeys = yamlkeys["blobs"]["field-batches"]["url"]
            azcopy_list(url_yamlkeys, read_yamlkeys, self.file_path)
            log.info(
                "Successfully listed blob contents and saved to %s", self.file_path
            )
        except Exception as e:
            log.error("Error listing blob contents: %s", e)
            raise

    def analyze_folders(self) -> Dict[str, Any]:
        """
        Analyzes the folder structure from the file listing blob contents.
        Logs the process and returns the organized batches.

        Returns:
            Dict[str, Any]: A dictionary with the organization of batches and their folder statuses.
        """
        log.info("Analyzing folders.")
        batches = {}
        try:
            with open(self.file_path, "r") as file:
                for line in file:
                    parts = line.split("/")
                    if len(parts) > 1:
                        batch_name = parts[0].split(" ")[1]
                        folder_name = parts[1]

                        if batch_name not in batches:
                            batches[batch_name] = {
                                "raws": False,
                                "metadata": False,
                                "preprocessed_jpgs": False,
                            }

                        if folder_name.lower() in [
                            "raws",
                            "metadata",
                            "preprocessed_jpgs",
                        ]:
                            batches[batch_name][folder_name.lower()] = True

            organized_batches = {key: batches[key] for key in sorted(batches)}
            log.info("Successfully analyzed and organized folders.")
            return organized_batches
        finally:
            log.info("Temporary file %s removed.", self.file_path)


class ReportWriter:
    """
    Manages the generation and writing of reports regarding missing batch information.
    """

    @staticmethod
    def write_missing_batch_info(
        batches: Dict[str, Any], output_file_path: str
    ) -> None:
        """
        Writes a report about missing batch folders to a specified file.
        Logs the process of writing this information.

        Args:
            batches (Dict[str, Any]): Dictionary containing the batches and their folder statuses.
            output_file_path (str): Path to the output file where the report will be written.
        """
        log.info("Writing missing batch info to %s.", output_file_path)
        try:
            with open(output_file_path, "w") as output_file:
                for batch, folders in batches.items():
                    if not all([folders["raws"], folders["metadata"]]):
                        if not folders["raws"]:
                            output_file.write(f"{batch}/raws\n")
                        if not folders["metadata"]:
                            output_file.write(f"{batch}/metadata\n")
            log.info("Successfully wrote missing batch info.")
        except Exception as e:
            log.error("Error writing missing batch info: %s", e)
            raise


def main(cfg: DictConfig) -> None:
    """
    Main function to orchestrate blob storage operations and report writing.
    Logs the start and completion of the main process.

    Args:
        cfg (DictConfig): A configuration object containing all necessary configurations.
    """
    log.info("Main process started with task: %s", cfg.general.task)
    try:
        file_path = "./tempoutput.txt"
        yaml = cfg.pipeline_keys
        output_file_path = cfg.report.missing_batch_folders

        blob_handler = BlobStorageHandler(yaml, file_path)
        blob_handler.list_blob_contents()
        batches = blob_handler.analyze_folders()

        ReportWriter.write_missing_batch_info(batches, output_file_path)
        log.info("Main process completed successfully.")
    except Exception as e:
        log.error("Main process encountered an error: %s", e)
        raise
