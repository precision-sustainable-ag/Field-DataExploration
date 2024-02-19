import yaml
import csv
import pandas as pd
def read_yaml(path):
    try :
        with open(path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except Exception as e :
        raise FileNotFoundError(f"File does not exist : {path}")

def read_csv_as_list(path):
    try :
            csv_reader = pd.read_csv(path)
            # Return as dataframe
            return csv_reader
    except Exception as e :
        raise FileNotFoundError(f"File does not exist : {path}")
