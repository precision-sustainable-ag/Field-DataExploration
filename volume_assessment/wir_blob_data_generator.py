#!/usr/bin/env python3

import os,sys
import yaml
from datetime import datetime
import asyncio
import pandas as pd 
from azure.storage.blob.aio import BlobServiceClient
from tqdm.asyncio import tqdm_asyncio
from from_root import from_root, from_here
sys.path.append(str(from_root('logging')))
from data_logger import log_images

async def get_blob_metrics(account_url, sas_token, container_name):
    """
    Uses container client to return detailed metrics for jpg and raw files in the container.

    Returns:
        images_details (list): List of dictionaries with details for each image.
    """
    images_details = []

    async with BlobServiceClient(
        account_url=account_url, credential=sas_token
    ) as blob_service_client:
        container_client = blob_service_client.get_container_client(container_name)

        async for blob in tqdm_asyncio(container_client.list_blobs()):

            image_detail = {
                "name": blob.name, # blob name 
                "memory_mb": float(blob.size / pow(1024, 2)), # convert kb to mb
                "container": blob.container, # get container name
                "creation_time_utc": blob.creation_time, # get creation time
            }
            images_details.append(image_detail)

    return images_details

async def main():

    __auth_config_name = 'authorized_keys.yaml'
    with open((os.path.join(str(from_root('')),__auth_config_name)), 'r') as file:
        __auth_config_data = yaml.safe_load(file)

    container_name = "weedsimagerepo"
    sas_token = __auth_config_data["blobs"][container_name]["sas_token"] 
    account_url = __auth_config_data["blobs"][container_name]["url"]

    # Get data from Blob servers
    images_details = await get_blob_metrics(account_url, sas_token, container_name)
    # Convert the list of dictionaries to a Pandas DataFrame
    time_stamp = datetime.utcnow().strftime("%Y-%m-%d")

    log_images(time_stamp,container_name,images_details)

asyncio.run(main())