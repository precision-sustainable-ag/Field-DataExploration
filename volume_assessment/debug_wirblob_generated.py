#!/usr/bin/env python3

import os,sys
import yaml
import asyncio
from azure.storage.blob import BlobServiceClient
from tqdm import tqdm
from from_root import from_root, from_here
import pandas as pd
import matplotlib.pyplot as plt 

def get_blob_metrics(account_url, sas_token, container_name):
    """Uses container client to return a list of jpg files and a list of raw files present in the container

    Returns:
        jpg_files (list): list of present jpg files in container
        raw_files (list): list of present raw files in container
        total_memory_jpg (float): total (Gb) memory for JPG images
        total_memory_raw (float): total (Gb) memory for .ARW images
    """
    jpg_files = set()
    raw_files = set()
    total_memory_jpg = 0
    total_memory_raw = 0

    with BlobServiceClient(
        account_url=account_url, credential=sas_token
    ) as blob_service_client:
        container_client = blob_service_client.get_container_client(container_name)
 
        for blob in tqdm(container_client.list_blobs()):
            base_name, extension = os.path.splitext(blob.name)
            blob_client = container_client.get_blob_client(blob.name)
            blob_properties = blob_client.get_blob_properties()
            if extension.lower() == ".jpg":
                jpg_files.add(str(blob.name))
                total_memory_jpg += (blob_properties.size / pow(1024,3))
            elif extension.lower() == ".arw":
                raw_files.add(str(blob.name))
                total_memory_raw += (blob_properties.size / pow(1024,3))

    return list(jpg_files), list(raw_files), total_memory_jpg, total_memory_raw

def main():

    __auth_config_name = 'authorized_keys.yaml'
    with open((os.path.join(str(from_root('')),__auth_config_name)), 'r') as file:
        __auth_config_data = yaml.safe_load(file)

    container_name = "weedsimagerepo"
    sas_token = __auth_config_data["blobs"][container_name]["sas_token"] 
    account_url = __auth_config_data["blobs"][container_name]["url"]

    jpg_imgs_names, raw_imgs_names, total_memory_jpg, total_memory_raw = get_blob_metrics(account_url, sas_token, container_name)
    print(f"Total Storage used for JPG : {total_memory_jpg} GB")
    print(f"Total Storage used for ARW : {total_memory_raw} GB")

main()