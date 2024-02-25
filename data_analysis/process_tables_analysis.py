#!/usr/bin/env python3
import os, sys
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import logging

from utils.utils import read_csv_as_df
log = logging.getLogger(__name__)

from process_blob_analysis import BlobTablePreProcessing
"""
    Execute this script : python FIELD_REPORT.py general.task=[process_tables_analysis]
"""
class WIRTablesPreProcessing:
    """
    A class designed to pre-process wircovercropsmeta, wircropsmeta, wirweedsmeta and wirmastermeta to create a complete table.

    This reads respective csv files, performs data cleaning and merging.

    Attributes:
        __auth_config_data (dict): Configuration data containing Azure Blob Storage credentials,
                                including account URLs and SAS tokens for each container.

    Methods:
        __init__(cfg): Initializes the BlobMetricExporter instance with configuration from a DictConfig object.
        preprocess_imgrefs(blobs_csv,imageref_df): Inputs blob metric table and imageref table as pandas.DataFrame and exports pre-processed tables
    """
    def __init__(self, cfg: DictConfig) -> None:
        ## append filenames of relevant csv tables
        self.tables_dir = cfg.data.tablesdir
        wircovercropsmeta_fname = "wircovercropsmeta_table_metrics.csv"
        wircropsmeta_fname = "wircropsmeta_table_metrics.csv"
        wirweedsmeta_fname = "wirweedsmeta_table_metrics.csv"
        wirmastermeta_fname = "wirmastermeta_table_metrics.csv"

        ## get csv data 
        self.wircovercropsmeta_df = read_csv_as_df(os.path.join(self.tables_dir,wircovercropsmeta_fname))
        del self.wircovercropsmeta_df[self.wircovercropsmeta_df.columns[0]]
        self.wircovercropsmeta_df = self.wircovercropsmeta_df.drop(['RowKey',"PartitionKey","Affiliation"],axis=1)

        self.wircropsmeta_df = read_csv_as_df(os.path.join(self.tables_dir,wircropsmeta_fname))
        del self.wircropsmeta_df[self.wircropsmeta_df.columns[0]]
        self.wircropsmeta_df = self.wircropsmeta_df.drop('RowKey',axis=1)

        self.wirweedsmeta_df = read_csv_as_df(os.path.join(self.tables_dir,wirweedsmeta_fname))
        del self.wirweedsmeta_df[self.wirweedsmeta_df.columns[0]]
        self.wirweedsmeta_df = self.wirweedsmeta_df.drop('RowKey',axis=1)

        self.wirmastermeta_df = read_csv_as_df(os.path.join(self.tables_dir,wirmastermeta_fname))
        del self.wirmastermeta_df[self.wirmastermeta_df.columns[0]]
        self.wirmastermeta_df.rename(columns={"RowKey": "MasterRefID"},inplace=True)
        self.wirmastermeta_df = self.wirmastermeta_df.drop('WeedsOrCrops',axis=1)

        ## append filename of merged csv file
        self.processed_tables_dir = cfg.data.processed_datadir
        wirmergedtable_fname = "merged_blobs_refs.csv"
        self.wirmergedtable_df = read_csv_as_df(os.path.join(self.processed_tables_dir,wirmergedtable_fname))
        del self.wirmergedtable_df[self.wirmergedtable_df.columns[0]]

        self.process_wir_tables()

    def process_wir_tables(self):

        processed_table = pd.merge(self.wirmastermeta_df,self.wircovercropsmeta_df,how="left",on=["MasterRefID","CloudCover","GroundResidue","GroundCover"])
        csv_path = Path(self.processed_tables_dir, "debug_processed.csv")
        processed_table.to_csv(csv_path)

def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    exporter = WIRTablesPreProcessing(cfg)
    log.info(f"{cfg.general.task} completed.")