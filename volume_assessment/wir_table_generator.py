from azure.core.credentials import AzureSasCredential
from azure.data.tables import TableServiceClient
import asyncio
import yaml
import pandas as pd
import os, sys
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
from from_root import from_root, from_here
sys.path.append(str(from_root('logging')))
from data_logger import log_images

def get_table_csv(account_url, sas_token, table_name):

    try:
        with TableServiceClient(
            endpoint=account_url, credential=AzureSasCredential(sas_token)
        ) as table_service_client:
            table_client = table_service_client.get_table_client(table_name=table_name)
            # Fetch all entities from the specified table
            entities = list(table_client.list_entities())
            # Convert entities to a list of dictionaries (assuming entities are not empty)
            data = [entity for entity in entities]
            # Convert list of dictionaries to DataFrame
            table_data = pd.DataFrame(data)
        return table_data

    except Exception as error :
        print(f"Error! Check {table_name} authorization parameters")

def main():

    __auth_config_name = 'authorized_keys.yaml'
    with open((os.path.join(str(from_root('keys')),__auth_config_name)), 'r') as file:
        auth_config_data = yaml.safe_load(file)
    time_stamp = datetime.utcnow().strftime("%Y-%m-%d")
    for table in auth_config_data["tables"] :
        table_name = table
        sas_token = auth_config_data["tables"][table_name]["sas_token"] 
        account_url = auth_config_data["tables"][table_name]["url"]
        # Get data from Blob servers
        table_data = get_table_csv(account_url, sas_token, table_name)
        # Create log if there is any table data
        if not table_data.empty:
            log_images(time_stamp,table_name,table_data)
    

main()