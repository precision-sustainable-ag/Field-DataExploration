#!/usr/bin/env python3
# fmt: off
# isort: off
import os
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import logging

from utils.utils import read_csv_as_df
log = logging.getLogger(__name__)

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
        self.wircovercropsmeta_df = self.wircovercropsmeta_df.drop(['RowKey',"PartitionKey","Affiliation"],axis=1)

        self.wircropsmeta_df = read_csv_as_df(os.path.join(self.tables_dir,wircropsmeta_fname))
        self.wircropsmeta_df = self.wircropsmeta_df.drop('RowKey',axis=1)
        # [Q: Check CropTypr for N/A values? fill nan if not required ]

        self.wirweedsmeta_df = read_csv_as_df(os.path.join(self.tables_dir,wirweedsmeta_fname))
        self.wirweedsmeta_df = self.wirweedsmeta_df.drop('RowKey',axis=1)

        self.wirmastermeta_df = read_csv_as_df(os.path.join(self.tables_dir,wirmastermeta_fname))
        self.wirmastermeta_df.rename(columns={"RowKey": "MasterRefID"},inplace=True)
        self.wirmastermeta_df = self.wirmastermeta_df.drop('WeedsOrCrops',axis=1)

        ## append filename of merged csv file
        self.processed_tables_dir = cfg.data.processed_datadir
        self.wirmergedtable_fname = "merged_blobs_tables_metadata.csv"
        self.wirmergedtable_df = read_csv_as_df(os.path.join(self.processed_tables_dir,self.wirmergedtable_fname))

        self.process_wir_tables()

    def process_wir_tables(self):

        # merge MasterMeta and CoverCrops
        processed_table = pd.merge(self.wirmastermeta_df,self.wircovercropsmeta_df,how="outer",on=["MasterRefID","CloudCover","GroundResidue","GroundCover"])
        # merge [ MasterMeta , CoverCrops] and Crops
        processed_table = pd.merge(processed_table,self.wircropsmeta_df,how="outer",on=["PartitionKey","MasterRefID"])
        processed_table.rename(columns={"CropName":"CropType"},inplace=True) # everythin fine till here
        # merge [MasterMeta , CoverCrops , Crops] and Weeds
        processed_table = pd.merge(processed_table,self.wirweedsmeta_df,how="outer",on=["PartitionKey","MasterRefID"],suffixes=('_01', '_02'))
        # merge [MasterMeta , CoverCrops , Crops, Weeds] and Blob Metadata
        processed_table = pd.merge(self.wirmergedtable_df,processed_table,how="outer",on=["PartitionKey","MasterRefID"])
        processed_table = processed_table[processed_table['name'].notna()]

        # Create Species Column
        processed_table["Species"] = None
        processed_table["Species"] = processed_table["Species"].fillna(processed_table["WeedType"])
        processed_table = processed_table.drop('WeedType',axis=1)

        processed_table["Species"] = processed_table["Species"].fillna(processed_table["CoverCropSpecies"])
        processed_table = processed_table.drop('CoverCropSpecies',axis=1)

        processed_table["Species"] = processed_table["Species"].fillna(processed_table["CropType_01"])
        processed_table = processed_table.drop('CropType_01',axis=1)

        processed_table.rename(columns={"CropType_02": "CropTypeSecondary"},inplace=True)
        # Create Height Column
        processed_table["Height"] = None
        processed_table["Height"] = processed_table["Height"].fillna(processed_table["Height_01"])
        processed_table = processed_table.drop('Height_01',axis=1)

        processed_table["Height"] = processed_table["Height"].fillna(processed_table["Height_02"])
        processed_table = processed_table.drop('Height_02',axis=1)

        # Create SizeClass Column
        processed_table["SizeClass"] = None
        processed_table["SizeClass"] = processed_table["SizeClass"].fillna(processed_table["SizeClass_01"])
        processed_table = processed_table.drop('SizeClass_01',axis=1)

        processed_table["SizeClass"] = processed_table["SizeClass"].fillna(processed_table["SizeClass_02"])
        processed_table = processed_table.drop('SizeClass_02',axis=1)

        processed_table["SizeClass"] = processed_table["SizeClass"].replace({'Large': 'LARGE','Medium': 'MEDIUM','Small':'SMALL',"3": "LARGE", "2": "MEDIUM", "1": "SMALL"})
        csv_path = Path(self.processed_tables_dir, self.wirmergedtable_fname)
        processed_table.to_csv(csv_path, index=False)

def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    exporter = WIRTablesPreProcessing(cfg)
    log.info(f"{cfg.general.task} completed.")
