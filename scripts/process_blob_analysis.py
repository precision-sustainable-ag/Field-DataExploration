#!/usr/bin/env python3
# fmt: off
# isort: off
import os, sys
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import logging

from utils.utils import read_csv_as_df
log = logging.getLogger(__name__)
"""
    Execute this script : python FIELD_REPORT.py general.task=[process_blob_analysis]
"""
class BlobTablePreProcessing:
    """
    A class designed to pre-process wirimagerefs.csv and weedsimagerepo_blob_metrics.csv.

    This reads respective csv files, performs data cleaning and merging.

    Exports merged_blobs_refs.csv and missing_blobs_metadata.csv

    Attributes:
        __auth_config_data (dict): Configuration data containing Azure Blob Storage credentials,
                                including account URLs and SAS tokens for each container.

    Methods:
        __init__(cfg): Initializes the BlobMetricExporter instance with configuration from a DictConfig object.
        preprocess_imgrefs(blobs_csv,imageref_df): Inputs blob metric table and imageref table as pandas.DataFrame and exports pre-processed tables
    """
    def __init__(self, cfg: DictConfig) -> None:

        blob_fname = "weedsimagerepo_blob_metrics.csv"
        table_fname = "wirimagerefs_table_metrics.csv"
        self.processed_blob_ref_fname = 'merged_blobs_tables_metadata.csv'
        self.missing_blob_fname = 'missing_blobs_metadata.csv'
        self.blob_table_dir = cfg.data.blobsdir
        self.refs_table_dir = cfg.data.tablesdir
        self.processed_datadir = cfg.data.processed_datadir
        Path(self.processed_datadir).mkdir(exist_ok=True, parents=True)
        self.blobs_csv = read_csv_as_df(os.path.join(self.blob_table_dir,blob_fname))

        self.imagerefs_csv = read_csv_as_df(os.path.join(self.refs_table_dir,table_fname))
        # Update original imagerefs table
        self.preprocess_imgrefs(self.blobs_csv,self.imagerefs_csv)
  
    def preprocess_imgrefs(self,blobs_csv,imageref_df):
        # Get image names from urls and adding a new column
        imageref_df["name"] = imageref_df["ImageURL"].apply(lambda url:os.path.basename(url))
        imageref_df["Wirimagerefs_rowkey"] = imageref_df["RowKey"]
        imageref_df["Wirimagerefs_timestamp"] = imageref_df["Timestamp"]
        imageref_df = imageref_df.drop('Timestamp',axis=1)
        # Left join on key = "name"
        processed_blobs = pd.merge(blobs_csv,imageref_df,on="name",how="left",left_index=False,right_index=False)
        # Get images that don't have MasterRefID
        missing_rows = processed_blobs.query('MasterRefID != MasterRefID')
        # Delete rows with all nan values
        missing_rows = missing_rows.dropna(axis=1, how='all')
        # remove values with missing MasterRefID
        processed_blobs = processed_blobs[processed_blobs['MasterRefID'].notna()]
        # removing RowKey from merged csv
        processed_blobs = processed_blobs.drop('RowKey',axis=1)
        if processed_blobs.empty:
            log.error(f"processed_blobs df is empty, Not saving!")
        else:
            # Save to csv
            csv_path = Path(self.processed_datadir, self.processed_blob_ref_fname)
            processed_blobs.to_csv(csv_path, index=False)
            log.info(f"Exported processed_blobs to {csv_path}.")
            csv_path = Path(self.processed_datadir, self.missing_blob_fname)
            missing_rows.to_csv(csv_path, index=False)
            log.info(f"Exported missing_rows to {csv_path}.")
            

def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    exporter = BlobTablePreProcessing(cfg)
    log.info(f"{cfg.general.task} completed.")
