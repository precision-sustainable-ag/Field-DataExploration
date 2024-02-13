#!/usr/bin/env python3
import os
from datetime import datetime
from from_root import from_here
import csv
from pathlib import Path

def log_memory(time_stamp,container_name:str,jpg_size_data:float,raw_size_data:float):

    # if csv file doesn't exist, create one
    memory_log_dir = from_here("blob_memory_logs")
    Path(memory_log_dir).mkdir(parents=True, exist_ok=True)
    memory_log_file = memory_log_dir / "memory_log.csv"

    # Create File is not exists
    if not os.path.exists(memory_log_file):
        i_fields = ['Timestamp', 'Container Name','JPG Size (Gb)','ARW Size (Gb)']
        with open(memory_log_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=i_fields)
            writer.writeheader()
    # Create sample data log query
    i_query = {'Timestamp': time_stamp, 
               'Container Name': container_name,
               'JPG Size (Gb)': jpg_size_data,
               'ARW Size (Gb)': raw_size_data}

    with open(memory_log_file, 'a', newline='') as file: 
        writer = csv.DictWriter(file, fieldnames = i_query)
        writer.writerow(i_query)

    print("Logged Blob Metrics",i_query)
