# Chapter 3: Azure Data Acquisition

Welcome back! In [Chapter 1: Configuration Management](01_configuration_management_.md), we learned how the project gets its instructions (its "settings") from YAML files using Hydra. Then, in [Chapter 2: Pipeline Task Runner](02_pipeline_task_runner_.md), we saw how the `main.py` script acts as a "task manager," reading the list of steps from the configuration and running the appropriate code for each step.

Now, it's time to dive into one of the crucial *first* steps that the Task Runner might execute: getting the raw data the project needs to work with. This data is stored in Microsoft Azure's cloud storage services.

### What is Azure Data Acquisition?

Imagine your project is a detective agency, and the data it needs is stored in various filing cabinets located in a massive warehouse (Azure). The first step in any investigation is to *acquire* the relevant files and records from those cabinets.

**Azure Data Acquisition** is the process of connecting to these cloud-based "filing cabinets" in Azure and fetching the raw lists of files (like images) and structured records (like notes or details about those images) that our project will analyze.

**The Central Use Case:** For this project, the most common data we need is:
1.  A list of all image files stored in Azure Blob Storage, along with some basic information about each file (like its name, size, and when it was uploaded).
2.  Structured data (like tags or classifications) related to these images, stored in Azure Table Storage.

This chapter explains how the project uses specific tasks to connect to these Azure services, download these lists and records, and save them locally so the rest of the pipeline can use them.

### Key Concepts

Before we look at the code, let's quickly understand the main Azure services involved:

1.  **Azure Blob Storage:** This is like a gigantic digital warehouse for storing all sorts of files (images, videos, documents, etc.). In our case, it holds the raw image files. The project doesn't download *all* the images right away (that would take too much space and time!). Instead, it focuses on getting a *list* of the files and their basic properties. Think of it as getting an index of all the files in a set of boxes, not opening every box and looking at its contents.

2.  **Azure Table Storage:** This service is designed for storing structured, non-relational data. It's like a spreadsheet in the cloud. We use it to store metadata or details associated with the files in Blob Storage. For example, a table might contain entries saying "Image ABC has a tag of 'weed'" or "Image XYZ was captured at location Alpha." The project downloads the *entire content* of these relevant tables.

3.  **Authentication:** To access your private data in Azure, the project needs permission. It proves it has permission using special credentials, like **Shared Access Signatures (SAS) tokens**. These are like temporary keys that grant specific, limited access to your storage. The project reads where to find these keys from the configuration (specifically `cfg.pipeline_keys`) and uses them to connect to Azure. We won't go into the details of generating these keys here, as that's covered in [Chapter 10: Azure Authentication Keys](10_azure_authentication_keys_.md).

### How to Use Azure Data Acquisition

Just like any other step in the pipeline, you don't directly call the Azure acquisition code yourself. You tell the [Pipeline Task Runner](02_pipeline_task_runner_.md) to run it by including the relevant task names in the `pipeline` list within your `conf/config.yaml` file.

The two main tasks responsible for data acquisition are:

*   `wir_table_generator`: This task connects to Azure Table Storage and downloads table data.
*   `wir_blob_data_generator`: This task connects to Azure Blob Storage and downloads blob (file) metrics (like name, size, creation time).

To make sure the project acquires data from Azure when you run it, you need to include these task names in your `pipeline` list.

Here's a simplified example of how your `conf/config.yaml` might look if you want to start by acquiring data:

```yaml
# conf/config.yaml (Snippet showing data acquisition tasks)

# ... other settings ...

pipeline:
    - wir_table_generator     # First, get the data from Azure Tables
    - wir_blob_data_generator # Second, get the list of files from Azure Blobs
    # - process_data          # (Optional) Next steps like processing...
    # - report                # (Optional) Then reporting...

# ... other settings ...
```

When you run `python main.py` with this configuration, the [Pipeline Task Runner](02_pipeline_task_runner_.md) will see `wir_table_generator` and `wir_blob_data_generator` in the list. It will then find and run the code associated with these names (which lives in `src/wir_table_generator.py` and `src/wir_blob_data_generator.py`).

### How It Works Under the Hood

Let's see the simplified process when you trigger one of these data acquisition tasks:

```mermaid
sequenceDiagram
    participant User
    participant MainPy as main.py (Task Runner)
    participant TaskCode as Task Code (e.g., wir_table_generator.py)
    participant AuthKeys as keys/authorized_keys.yaml
    participant Config as Configuration (cfg)
    participant Azure as Azure Storage
    participant LocalFile as Local Data Files

    User->MainPy: Run script (python main.py)
    MainPy->Config: Load configuration (including cfg.pipeline)
    MainPy->MainPy: Get task list from cfg.pipeline
    loop For each acquisition task (e.g., wir_table_generator)
        MainPy->TaskCode: Run task code (pass cfg)
        TaskCode->Config: Get settings from cfg (e.g., cfg.pipeline_keys, cfg.paths.tablesdir)
        TaskCode->AuthKeys: Read authorization keys from file path
        TaskCode->Azure: Connect using keys & settings
        Azure-->TaskCode: Return requested data (table rows or blob list)
        TaskCode->TaskCode: Process data (e.g., put in DataFrame)
        TaskCode->LocalFile: Save data as CSV
        LocalFile-->TaskCode: Confirm save
    end
    MainPy: Acquisition tasks completed!
```

As you can see, the Task Runner simply calls the specific code responsible for the task. This task code then uses the `cfg` object (which contains *all* the configuration, including the path to the keys file and where to save outputs) to perform its job.

Let's look at simplified snippets from the actual files responsible:

#### Acquiring Table Data (`src/wir_table_generator.py`)

This file contains the logic to fetch data from Azure Table Storage.

```python
# src/wir_table_generator.py (Simplified Snippet)
import logging
from pathlib import Path
import pandas as pd
from azure.data.tables import TableServiceClient # Azure library for Tables
from omegaconf import DictConfig
from utils.utils import read_yaml # Helper to read keys file

log = logging.getLogger(__name__)

class TableExporter:
    def __init__(self, cfg: DictConfig) -> None:
        # Get the path to the keys file from config and read it
        self.__auth_config_data = read_yaml(cfg.pipeline_keys)
        # Get the directory to save tables from config
        self.tables_dir = cfg.paths.tablesdir
        # Make sure the directory exists
        Path(self.tables_dir).mkdir(exist_ok=True, parents=True)

    def get_table_data(self, account_url, sas_token, table_name):
        # --- Simplified Azure Connection ---
        try:
            # Use Azure library to connect with URL and SAS token
            table_service_client = TableServiceClient(
                endpoint=account_url, credential=sas_token # SAS token used directly here for simplicity, actual code uses AzureSasCredential
            )
            table_client = table_service_client.get_table_client(table_name=table_name)
            # List all rows (entities) in the table
            entities = []
            for i in table_client.list_entities():
                 # Process each entity... (simplified)
                 entities.append(i)
            return entities
        except Exception as error:
            log.exception(f"Error connecting to {table_name}")
            return [] # Return empty list on error

    def get_table_csv(self):
        # Loop through each table defined in the keys file
        for table_name in self.__auth_config_data["tables"]:
            # Get URL and SAS token for this specific table
            sas_token = self.__auth_config_data["tables"][table_name]["sas_token"]
            account_url = self.__auth_config_data["tables"][table_name]["url"]

            # Call the function to get data from Azure
            table_data = self.get_table_data(account_url, sas_token, table_name)

            if table_data:
                # Put the data into a pandas DataFrame
                df = pd.DataFrame(table_data)
                # Define the save path using the directory from config
                csv_path = Path(self.tables_dir, f"{table_name}_table_metrics.csv")
                # Save the DataFrame to a CSV file
                df.to_csv(csv_path, index=False)
                log.info(f"Exported {table_name} data to {csv_path}")
            else:
                 log.warn(f"{table_name} data is empty or connection failed.")


def main(cfg: DictConfig) -> None:
    # This is the entry point called by the Task Runner
    log.info(f"Starting Azure Table data acquisition")
    exporter = TableExporter(cfg) # Create an instance, passing the config
    exporter.get_table_csv()    # Run the method to fetch and save data
    log.info(f"Azure Table data acquisition completed.")

```

**Explanation:**

*   The `main(cfg: DictConfig)` function is the entry point, receiving the `cfg` object from the [Pipeline Task Runner](02_pipeline_task_runner_.md).
*   Inside `main`, it creates an instance of the `TableExporter` class, passing `cfg`.
*   The `TableExporter`'s `__init__` method reads the `authorized_keys.yaml` file (whose path is found in `cfg.pipeline_keys`) to get the Azure connection details. It also gets the output directory from `cfg.paths.tablesdir`.
*   The `get_table_csv` method loops through the tables listed in the keys file. For each table, it calls `get_table_data`.
*   `get_table_data` uses the Azure client library (`azure.data.tables`) to connect to Azure using the provided URL and SAS token and fetches the data.
*   Finally, `get_table_csv` takes the fetched data, turns it into a pandas DataFrame, and saves it as a CSV file in the directory specified by `cfg.paths.tablesdir`.

#### Acquiring Blob Metrics (`src/wir_blob_data_generator.py`)

This file does a similar job but for Azure Blob Storage, fetching a *list* of files and their details.

```python
# src/wir_blob_data_generator.py (Simplified Snippet)
import logging
from pathlib import Path
import pandas as pd
from azure.storage.blob import BlobServiceClient # Azure library for Blobs
from omegaconf import DictConfig
from utils.utils import read_yaml # Helper to read keys file

log = logging.getLogger(__name__)

class BlobMetricExporter:
    def __init__(self, cfg) -> None:
        # Get the path to the keys file from config and read it
        self.__auth_config_data = read_yaml(cfg.pipeline_keys)
        # Get the directory to save blob lists from config
        self.blobs_dir = cfg.paths.blobsdir
        # Make sure the directory exists
        Path(self.blobs_dir).mkdir(exist_ok=True, parents=True)


    def get_blob_metrics(self, account_url, sas_token, container_name):
        # --- Simplified Azure Connection ---
        try:
            images_details = []
            # Use Azure library to connect with URL and SAS token
            blob_service_client = BlobServiceClient(
                account_url=account_url, credential=sas_token
            )
            container_client = blob_service_client.get_container_client(container_name)

            # List all blobs (files) in the container
            for blob in container_client.list_blobs():
                # Extract relevant details for each blob (simplified)
                image_detail = {
                    "name": blob.name,
                    "memory_mb": float(blob.size / pow(1024, 2)),
                    "container": blob.container,
                    "creation_time_utc": blob.creation_time,
                }
                images_details.append(image_detail)

            return images_details # Return the list of blob details

        except Exception as error:
            log.exception(f"Error connecting to container {container_name}")


    def get_blob_csv(self, container_name="weedsimagerepo"):
        # Get SAS token for the container and account URL from keys file
        sas_token = self.__auth_config_data["blobs"][container_name]["sas_token"]
        account_url = self.__auth_config_data["blobs"]["account_url"] # Note: URL might be nested differently in actual keys

        # Call the function to get blob metrics from Azure
        images_details = self.get_blob_metrics(account_url, sas_token, container_name)
        if images_details:
            # Put the details into a pandas DataFrame
            df = pd.DataFrame(images_details)
            # Define the save path using the directory from config
            csv_path = Path(self.blobs_dir, f"{container_name}_blob_metrics.csv")
            # Save the DataFrame to a CSV file
            df.to_csv(csv_path, index=False)
            log.info(f"Exported {container_name} data to {csv_path}")
        else:
             log.warn(f"{container_name} data is empty or connection failed.")


def main(cfg: DictConfig) -> None:
    # This is the entry point called by the Task Runner
    log.info(f"Starting Azure Blob data acquisition")
    exporter = BlobMetricExporter(cfg) # Create an instance, passing the config
    # Note: Container name might be configured elsewhere or handled dynamically
    exporter.get_blob_csv(container_name="weedsimagerepo") # Run the method to fetch and save
    log.info(f"Azure Blob data acquisition completed.")

```

**Explanation:**

*   This file works very similarly to `wir_table_generator.py`. The `main` function is the entry point, creating a `BlobMetricExporter` instance.
*   The `__init__` method reads the `authorized_keys.yaml` file and gets the output directory (`cfg.paths.blobsdir`) from the configuration.
*   The `get_blob_csv` method gets the connection details (URL, SAS token) for the specified blob container from the keys file.
*   `get_blob_metrics` uses the Azure client library (`azure.storage.blob`) to connect and then uses `container_client.list_blobs()` to get a list of all files (blobs) in that container. It extracts the name, size, and creation time for each.
*   Finally, `get_blob_csv` takes this list of details, creates a pandas DataFrame, and saves it as a CSV file in the directory from `cfg.paths.blobsdir`.

After these tasks run, you will find new CSV files in the directories specified by `cfg.paths.tablesdir` and `cfg.paths.blobsdir`. These files contain the raw lists of files and table records acquired from Azure.

### Summary

In this chapter, we learned about **Azure Data Acquisition**, the essential first step for getting data into the `Field-DataExploration` project. We saw that this involves connecting to:

*   **Azure Blob Storage** to get lists and basic metrics of image files.
*   **Azure Table Storage** to get structured metadata records.

These connections are made using credentials (like SAS tokens) read from a separate keys file, whose location is specified in the project's configuration (`cfg.pipeline_keys`).

We also learned how these tasks fit into the overall flow: they are specific tasks (`wir_table_generator`, `wir_blob_data_generator`) listed in `conf/config.yaml`'s `pipeline`, and the [Pipeline Task Runner](02_pipeline_task_runner_.md) is responsible for executing their code (found in `src/wir_table_generator.py` and `src/wir_blob_data_generator.py`) and providing them with the necessary configuration (`cfg`).

The output of these tasks are raw CSV files containing the acquired data saved locally. This data is now ready for the next steps in the pipeline!

Now that we have the raw data lists and records from Azure, the next logical step is to combine this information and get it into a usable format.

Let's move on to the next chapter where we'll explore **Data Integration & Preprocessing**.

[Data Integration & Preprocessing](04_data_integration___preprocessing_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)