import logging
from omegaconf import DictConfig
from azure.core.credentials import AzureSasCredential
from azure.data.tables import TableServiceClient
from pathlib import Path
import os
import pandas as pd
from typing import Set
from datetime import datetime
import csv

from utils.utils import (
    azcopy_list,
    download_azcopy,
    find_most_recent_csv,
    get_exif_data,
    read_csv_as_df,
    read_yaml,
		download_azcopy_multiple
)

log = logging.getLogger(__name__)

class Backup:
    """
    Backup data from 6 tables and weedsimagerepo blob in the weedsimagerepo container in azure storage
    """

    def __init__(self, cfg: DictConfig):
        self.yamlkeys = read_yaml(cfg.pipeline_keys)
    
    def list_azstorage_contents(self, blob_name) -> None:
        # Lists the contents of the blob storage
        log.info("Listing storage contents.")
        try:
            read_yamlkeys = self.yamlkeys["blobs"][blob_name]["sas_token"]
            url_yamlkeys = self.yamlkeys["blobs"][blob_name]["url"]
            azcopy_list(url_yamlkeys, read_yamlkeys, "./temp_" + blob_name + ".txt", blob_name)
            log.info("Storage contents successfully listed for: %s ", blob_name, exc_info=True)
        except Exception as e:
            log.error("Failed to list storage contents: %s", e, exc_info=True)
            raise
		
    def blob_filenames(self, blob_name) -> Set[str]:
        # Extracts filenames from the listed blob contents.
        files = set()
        with open("./temp_" + blob_name + ".txt", "r") as file:
            for line in file:
                filename = line.split(" ")[1].replace(";", "")
                files.add(filename)
        log.info("%d total images identified.", len(files))

        # Ensure the temporary file is removed after processing
        os.remove("./temp_" + blob_name + ".txt")
        return files

    def get_backedup_filenames(self, backup_dir) -> Set[str]:
        # Extracts filenames from backup directory
        files = set()
        path = Path(backup_dir)
        for file in path.iterdir():
            if file.is_file():
                files.add(file.name)
        log.info("%d total images identified in backup location.", len(files))
        return files
    
    def download_pending_blobs(self, blob_name, backup_dir, file_list):
        # Downloads files mentioned in file_list from azure blob storage
        log.info(f"Starting backup for blob:  {blob_name}")
        read_yamlkeys = self.yamlkeys["blobs"][blob_name]["sas_token"]
        url_yamlkeys = self.yamlkeys["blobs"][blob_name]["url"]
        backuppath = Path(backup_dir)
        
        # Splitting files into groups of 100 each as azcopy has a limitation on the parameter length
        split_file_list = [file_list[i:i+100] for i in range(0, len(file_list), 100)]
        for files in split_file_list:
            download_azcopy_multiple(url_yamlkeys, read_yamlkeys, backuppath, files, blob_name)

    def get_tables_list(self):
        # Reads authorized_keys.yaml file and generates a list of table names	
        tables = self.yamlkeys["tables"]
        return list(tables.keys())

    def download_table_contents(self, table_name, backup_path):
        # Downloads contents of each table as a csv file and saves them in the backup directory based on the backup date
        log.info(f"Starting backup for table:  {table_name}")
        read_yamlkeys = self.yamlkeys["tables"][table_name]["sas_token"]
        url_yamlkeys = self.yamlkeys["tables"][table_name]["url"]
        table_service_client = TableServiceClient(
            endpoint=url_yamlkeys, credential=AzureSasCredential(read_yamlkeys)
        )
        table_client = table_service_client.get_table_client(table_name=table_name)
        # Fetch all entities from the specified table and add timestamps
        table_data = []
        for i in table_client.list_entities():
            timestamp = str(i._metadata["timestamp"])
            i["Timestamp"] = timestamp
            table_data.append(i)

        if table_data:
            df_table_details = pd.DataFrame(table_data)
            # Export to CSV
            csv_path = Path(backup_path, f"{table_name}_table_backup.csv")
            df_table_details.to_csv(csv_path, index=False)
            log.info(f"Exported {table_name} data to {csv_path}")
        else:
            log.warn(f"{table_name} data is empty, Not saving!")
        return table_data


def main(cfg: DictConfig) -> None:
    # Main function to orchestrate the data processing.
    log.info(f"Starting {cfg.general.task}")
    try:
        backup = Backup(cfg)

        # setting backup location
        current_dir = os.path.abspath(os.path.dirname(__file__))
        parent_dir = os.path.dirname(current_dir)
        parent_dir = os.path.abspath('/mnt/research-projects/r/raatwell/longterm_images3')

        datestring = datetime.now().strftime("%Y-%m-%d")
        tables_backup_dir = "/".join(parent_dir.split("/") + ["weedsimagerepo", "tables", datestring])
        Path(tables_backup_dir).mkdir(exist_ok=True, parents=True)

        # getting list of tables from pipeline keys
        tables = backup.get_tables_list()
        # backing up each table
        for table in tables:
            backup.download_table_contents(table, tables_backup_dir)
        # get file names from azure blob
        
        blobs = ["weedsimagerepo"]
        for blob in blobs:
            backup.list_azstorage_contents(blob)
            azfiles = backup.blob_filenames(blob)

            # get backedup file names
            blobs_backup_dir = "/".join(parent_dir.split("/") + ["weedsimagerepo", "blobs", blob])
            Path(blobs_backup_dir).mkdir(exist_ok=True, parents=True)
            backedupfiles = backup.get_backedup_filenames(blobs_backup_dir)

            # get files that are not backed up
            pending_files = list(azfiles - backedupfiles)

            # download pending files
            blobs_upload_dir = "/".join(parent_dir.split("/") + ["weedsimagerepo", "blobs"])
            backup.download_pending_blobs(blob, blobs_upload_dir, pending_files)

    except Exception as e:
        log.error(f"An error occurred during {cfg.general.task}: {e}", exc_info=True)
