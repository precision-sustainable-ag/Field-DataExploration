#!/usr/bin/env python3
import os, sys
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import logging

import csv 
from utils.utils import read_csv_as_df
log = logging.getLogger(__name__)

class TablePreProcessing:

    def __init__(self, cfg: DictConfig) -> None:

        blob_fname = "weedsimagerepo_blob_metrics.csv"
        table_fname = "wirimagerefs_table_metrics.csv"
        self.processed_blob_ref_fname = 'merged_blobs_refs.csv'
        self.missing_blob_fname = 'missing_blobs_metadata.csv'
        self.blob_table_dir = cfg.data.blobsdir
        self.refs_table_dir = cfg.data.tablesdir
        self.processed_datadir = cfg.data.processed_datadir
        Path(self.processed_datadir).mkdir(exist_ok=True, parents=True)
        self.blobs_csv = read_csv_as_df(os.path.join(self.blob_table_dir,blob_fname))
        # removing index
        del self.blobs_csv[self.blobs_csv.columns[0]]

        self.imagerefs_csv = read_csv_as_df(os.path.join(self.refs_table_dir,table_fname))
        # removing index
        del self.imagerefs_csv[self.imagerefs_csv.columns[0]]
        # Update original imagerefs table
        self.preprocess_imgrefs(self.blobs_csv,self.imagerefs_csv)
  
    def preprocess_imgrefs(self,blobs_csv,imageref_df):
        # Get image names from urls and adding a new column
        imageref_df["name"] = imageref_df["ImageURL"].apply(lambda url:os.path.basename(url))
        # Left join on key = "name"
        processed_blobs = pd.merge(blobs_csv,imageref_df,on="name",how="left",left_index=False,right_index=False)
        # Get images that don't have MasterRefID
        missing_rows = processed_blobs.query('MasterRefID != MasterRefID')
        # Delete rows with all nan values
        missing_rows = missing_rows.dropna(axis=1, how='all')
        # Save to csv
        csv_path = Path(self.processed_datadir, self.processed_blob_ref_fname)
        processed_blobs.to_csv(csv_path)
        csv_path = Path(self.processed_datadir, self.missing_blob_fname)
        missing_rows.to_csv(csv_path)

def main(cfg: DictConfig) -> None:
    log.debug(f"Starting {cfg.general.task}")
    exporter = TablePreProcessing(cfg)
