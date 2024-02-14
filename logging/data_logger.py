#!/usr/bin/env python3
import os
from from_root import from_here, from_root
from pathlib import Path
import pandas as pd

table_log_dir = os.path.join(str(from_root("volume_assessment")),'data')
Path(table_log_dir).mkdir(parents=True, exist_ok=True)

def log_wirtable(time_stamp,container_name,image_list):
    """
        This image logger accepts arguments 
            Example :
                time_stamp (str) : datetime.utcnow().strftime("%Y-%m-%d")
                container_name (str) : "wirimagerepo"
                image_list (list[dict{}]) : [{
                                        "name": blob.name, # Get name 
                                        "memory_mb": float(blob.size / pow(1024, 2)), # mb
                                        "container": blob.container,
                                        "creation_time_utc": blob.creation_time,
                                    }]
    """

    # Create Dataframe for the table
    df_images_details = pd.DataFrame(image_list)
    # Export to csv 
    df_images_details.to_csv(f"{table_log_dir}/{container_name}_metrics_{time_stamp}.csv")