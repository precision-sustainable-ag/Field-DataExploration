import csv

import pandas as pd
import yaml


def read_yaml(path):
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")


def read_csv_as_df(path: str) -> pd.DataFrame:
    try:
        csv_reader = pd.read_csv(path)
        if "Unnamed: 0" in csv_reader.columns:
            csv_reader = csv_reader.drop(columns=["Unnamed: 0"])
        # Return as dataframe
        return csv_reader
    except Exception as e:
        raise FileNotFoundError(f"File does not exist : {path}")
