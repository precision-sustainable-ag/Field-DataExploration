# Chapter 10: Azure Authentication Keys

Welcome back to the final chapter of our tutorial series on the `Field-DataExploration` project! We've come a long way, learning how the project manages its settings ([Chapter 1: Configuration Management](01_configuration_management_.md)), runs different tasks automatically ([Chapter 2: Pipeline Task Runner](02_pipeline_task_runner_.md)), acquires data from Azure ([Chapter 3: Azure Data Acquisition](03_azure_data_acquisition_.md)), integrates and cleans it ([Chapter 4: Data Integration & Preprocessing](04_data_integration___preprocessing_.md)), enriches it with timestamps ([Chapter 5: Datetime Metadata Enrichment](05_datetime_metadata_enrichment_.md)), generates reports and plots ([Chapter 6: Reporting & Visualization](06_reporting___visualization_.md)), creates batches of images ([Chapter 7: Batch Generation](07_batch_generation_.md)), and performs image inspection ([Chapter 8: Image Inspection](08_image_inspection_.md)). Along the way, we've seen how the project uses helpful reusable code pieces called **Utility Functions** ([Chapter 9: Utility Functions](09_utility_functions_.md)).

Throughout these chapters, particularly when discussing data acquisition or interactions with Azure storage, we've mentioned that the project needs permission to access your data in the cloud. How does it get this permission? Through special **Azure Authentication Keys**.

### What are Azure Authentication Keys?

Imagine your data in Azure Blob and Table Storage is kept in a highly secure warehouse. To get anything in or out of that warehouse, you need a specific key or password.

**Azure Authentication Keys** are those special keys. They are sensitive pieces of information, like passwords or temporary access codes (often called **Shared Access Signatures, or SAS tokens**), that prove to Azure that the `Field-DataExploration` project is allowed to connect to your storage accounts and perform specific actions, like listing files, reading table entries, or downloading individual files.

**The Central Use Case:** The project absolutely *needs* these keys to access the raw data (lists of files, table records) stored in Azure. Without them, tasks like [Azure Data Acquisition](03_azure_data_acquisition_.md) or temporary image downloads for [Image Inspection](08_image_inspection_.md) simply cannot happen. The main goal here is to understand *what* these keys are and *how you provide them* to the project so it can access your cloud data.

### Why are these important?

*   **Security:** Keys ensure that only authorized applications (like this project, when you run it with your keys) can access your private data.
*   **Access Control:** Different keys or SAS tokens can grant different levels of access (read-only, write-only, read/write) to specific parts of your storage (like just one container or one table).
*   **Decoupling:** By keeping keys separate from the code, you can share or version the code without exposing your private credentials. You can also easily update keys without changing the project's logic.

### Key Concepts

*   **Sensitive Credentials:** These are secrets you must protect. They grant access to your cloud resources.
*   **SAS Tokens:** A common type of key used for Azure Storage. They are strings that are appended to the storage URL and grant time-limited, specific permissions (e.g., allow reading blobs in container 'X' for the next 24 hours).
*   **Separation from Code:** For security, these keys are *never* hardcoded directly into the Python scripts. They are stored in a separate file.
*   **Configuration Pointer:** The project's configuration ([Chapter 1: Configuration Management](01_configuration_management_.md)) tells the project *where* to find the file containing these keys.

### How to Use Azure Authentication Keys (As a User)

As a user of the `Field-DataExploration` project, you don't write code to *generate* or *handle* the keys themselves. Your responsibility is to:

1.  **Obtain** the necessary keys (SAS tokens) from your Azure environment. This is usually done through the Azure portal or Azure CLI.
2.  **Store** these keys securely in a specific file expected by the project.

The project looks for the keys in a YAML file. The path to this file is defined in the main configuration file, `conf/config.yaml`, using the `pipeline_keys` setting.

Look at the `conf/config.yaml` file snippet:

```yaml
# conf/config.yaml (Relevant Snippet)

# ... other settings ...

# This setting tells the project where to find the file with Azure keys
pipeline_keys: ${general.workdir}/keys/authorized_keys.yaml

# ... other sections ...
```

This line tells the project: "Go to the project's working directory (`${general.workdir}`), then look inside the `keys` folder, and find a file named `authorized_keys.yaml`. That file contains the Azure keys."

The project comes with a `keys` folder, and inside it, a `README.md` file that explains the required structure of the `authorized_keys.yaml` file you need to create.

Here's what that `keys/README.md` tells you (simplified):

```markdown
# Field-DataExploration

**Setting Private Authorization Keys**:

1. Create an authorized_keys.yaml file in the  ```keys/``` folder
  ```
  touch authorized_keys.yaml
  ```

2. Add blobs keys to the file

  Example :
  ```yaml
  blobs:
      # Account URL is needed for BlobServiceClient connection
      account_url: "https://<your-storage-account-name>.blob.core.windows.net/"
      # Each container the project needs access to is listed here
      <container-name-1>: # e.g., weedsimagerepo
          sas_token: "<SAS TOKEN FOR CONTAINER 1>" # Your actual, sensitive SAS token
      <container-name-2>: # e.g., field-batches (if needed for writing batches)
          write_sas_token: "<SAS TOKEN WITH WRITE PERMISSIONS FOR CONTAINER 2>" # Specific token for writing
          read_sas_token: "<SAS TOKEN WITH READ PERMISSIONS FOR CONTAINER 2>" # Specific token for reading
      # ... add other blob containers as needed ...
  ```

3. Add Table Keys to the file

  Example :
  ```yaml
  tables:
    # Each table the project needs access to is listed here
    <table-name-1>: # e.g., wirmastermeta
      url: "https://<your-storage-account-name>.table.core.windows.net/" # Table account URL
      sas_token: "<SAS TOKEN FOR TABLE 1>" # Your actual, sensitive SAS token

    <table-name-2>: # e.g., wirimagerefs
      url: "https://<your-storage-account-name>.table.core.windows.net/"
      sas_token: "<SAS TOKEN FOR TABLE 2>"

    # ... add other tables as needed ...
  ```
```

To "use" Azure Authentication Keys, you simply create the `authorized_keys.yaml` file in the `keys` folder and fill it in with the actual SAS tokens and URLs for your Azure storage accounts and containers/tables, following the structure shown above and in the `keys/README.md`.

**You must obtain these keys from your Azure administrator or the Azure portal yourself.** This project does not generate them for you; it only reads them from the file you provide.

**Example Input:** You would create `keys/authorized_keys.yaml` and put something like this in it (replacing the placeholders with your actual keys):

```yaml
# keys/authorized_keys.yaml (Example with placeholder keys)
blobs:
    account_url: "https://mystorageaccount.blob.core.windows.net/"
    weedsimagerepo:
        sas_token: "sv=2022-11-02&ss=b&srt=sco&sp=rl&se=2024-12-31T..." # This is a fake token!
    field-batches:
        write_sas_token: "sv=2022-11-02&ss=b&srt=sco&sp=rwl&se=2024-12-31T..." # Another fake token!
        read_sas_token: "sv=2022-11-02&ss=b&srt=sco&sp=rl&se=2024-12-31T..." # Another fake token!

tables:
  wirmastermeta:
    url: "https://mystorageaccount.table.core.windows.net/"
    sas_token: "sv=2022-11-02&ss=t&srt=sco&sp=rl&se=2024-12-31T..." # Fake token!

  wirimagerefs:
    url: "https://mystorageaccount.table.core.windows.net/"
    sas_token: "sv=2022-11-02&ss=t&srt=sco&sp=rl&se=2024-12-31T..." # Fake token!

# Add other necessary tables/blobs here following the structure
```

After you've created this file with your actual keys, the project, when run, will automatically find and read it because `conf/config.yaml` points to it.

### How It Works Under the Hood

Now, let's look at how the project *code* uses these keys after you've provided them in the `authorized_keys.yaml` file.

The flow goes like this:

1.  You run `python main.py`.
2.  [Hydra ([Chapter 1: Configuration Management](01_configuration_management_.md))] loads `conf/config.yaml`.
3.  Hydra resolves the `pipeline_keys` setting, determining the exact path to your `authorized_keys.yaml` file.
4.  Hydra passes the complete configuration, including the `pipeline_keys` path, to the main function (`run_FIELD_REPORT`) in `main.py` via the `cfg` object.
5.  The [Pipeline Task Runner ([Chapter 2: Pipeline Task Runner](02_pipeline_task_runner_.md))] starts executing the tasks listed in `cfg.pipeline` (e.g., `wir_table_generator`, `wir_blob_data_generator`, `create_batches`, `image_inspection`).
6.  Any task that needs to interact with Azure (like reading data or copying files) receives the `cfg` object.
7.  Inside these task scripts, they access `cfg.pipeline_keys` to get the path to the keys file.
8.  They then call the `read_yaml` utility function from `utils/utils.py` ([Chapter 9: Utility Functions](09_utility_functions_.md)), passing it the keys file path.
9.  `read_yaml` reads the `authorized_keys.yaml` file and returns its contents as a Python dictionary.
10. The task script uses the account URLs and SAS tokens from this dictionary to authenticate its connections when using the Azure client libraries (like `azure.data.tables` or `azure.storage.blob`) or the `azcopy` utility ([Chapter 7: Batch Generation](07_batch_generation_.md), [Chapter 8: Image Inspection](08_image_inspection_.md)).

Here's a sequence diagram showing the core parts of this process:

```mermaid
sequenceDiagram
    participant User
    participant MainPy as main.py
    participant HydraLib as Hydra Library
    participant ConfigYaml as conf/config.yaml
    participant AuthKeysYaml as keys/authorized_keys.yaml
    participant TaskScript as A Task (e.g. src/wir_table_generator.py)
    participant UtilsPy as utils/utils.py
    participant AzureClient as Azure Client Library
    participant AzureStorage as Azure Storage

    User->MainPy: Run script (python main.py)
    MainPy->HydraLib: Start (via @hydra.main)
    HydraLib->ConfigYaml: Load conf/config.yaml
    ConfigYaml-->HydraLib: Return config (includes pipeline_keys path)
    HydraLib->MainPy: Call run_FIELD_REPORT(cfg)
    MainPy->MainPy: Get task list from cfg.pipeline
    loop For each Azure-interacting task
        MainPy->TaskScript: Run main(cfg)
        TaskScript->TaskScript: Get keys file path from cfg.pipeline_keys
        TaskScript->UtilsPy: Call read_yaml(keys_path)
        UtilsPy->AuthKeysYaml: Read authorized_keys.yaml
        AuthKeysYaml-->UtilsPy: Return key data (dict)
        UtilsPy-->TaskScript: Return key data
        TaskScript->AzureClient: Initialize client with URL & SAS token from key data
        AzureClient->AzureStorage: Authenticated request (e.g., list blobs)
        AzureStorage-->AzureClient: Return data
        AzureClient-->TaskScript: Provide data
        TaskScript: Process data...
    end
    TaskScript-->MainPy: Task finished
    MainPy: All tasks completed!
```

Let's look at simplified code snippets to illustrate this:

#### Accessing the Keys File Path in `conf/config.yaml`

This is where the path is defined:

```yaml
# conf/config.yaml (Simplified Snippet)
# ... other defaults ...
pipeline_keys: ${general.workdir}/keys/authorized_keys.yaml
# ... other settings ...
```

As seen in [Chapter 1: Configuration Management](01_configuration_management_.md), this tells Hydra where to find the file relative to the project's working directory.

#### Reading the Keys File in a Task Script

Task scripts that need Azure access read the configuration (`cfg`) and then use the path within `cfg` to load the keys using the utility function:

```python
# src/wir_table_generator.py (Simplified Snippet)
import logging
from pathlib import Path
import pandas as pd
from azure.core.credentials import AzureSasCredential # Needed for authentication
from azure.data.tables import TableServiceClient # Azure library for Tables
from omegaconf import DictConfig
from tqdm import tqdm

from utils.utils import read_yaml # Import the utility function

log = logging.getLogger(__name__)

class TableExporter:
    def __init__(self, cfg: DictConfig) -> None:
        # Get the path to the keys file from config (cfg.pipeline_keys)
        keys_file_path = cfg.pipeline_keys
        # Use the utility function to read the keys file
        self.__auth_config_data = read_yaml(keys_file_path)

        # Get other settings from config
        self.tables_dir = cfg.paths.tablesdir
        Path(self.tables_dir).mkdir(exist_ok=True, parents=True)

    def get_table_data(self, account_url, sas_token, table_name):
        try:
            # Use the Azure library to connect, providing URL and SAS token
            table_service_client = TableServiceClient(
                endpoint=account_url, credential=AzureSasCredential(sas_token) # SAS token used here for authentication
            )
            table_client = table_service_client.get_table_client(table_name=table_name)
            # ... rest of the data fetching logic ...
            entities = []
            for i in table_client.list_entities():
                 entities.append(i)
            return entities

        except Exception as error:
            log.exception(f"Error accessing table {table_name}. Check keys.")
            return []

    def get_table_csv(self):
        # Loop through tables defined in the authorized_keys.yaml data
        for table_name in tqdm(self.__auth_config_data["tables"]):
            # Get the specific SAS token and URL for this table from the loaded data
            sas_token = self.__auth_config_data["tables"][table_name]["sas_token"]
            account_url = self.__auth_config_data["tables"][table_name]["url"]

            # Call the function that connects to Azure using these keys
            table_data = self.get_table_data(account_url, sas_token, table_name)
            # ... rest of saving logic ...

def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    exporter = TableExporter(cfg) # Pass cfg to the class
    exporter.get_table_csv()
    log.info(f"{cfg.general.task} completed.")

# ... rest of the file ...
```

**Explanation:**

*   The `main(cfg)` function receives the full configuration object (`cfg`).
*   Inside the `TableExporter` class's `__init__` method, it retrieves the path to the keys file using `cfg.pipeline_keys`.
*   It then calls `read_yaml(keys_file_path)` (the utility function) to load the contents of your `authorized_keys.yaml` file into the `self.__auth_config_data` variable.
*   Later, in the `get_table_csv` method, it loops through the tables defined in the loaded `__auth_config_data`. For each table, it extracts the specific `sas_token` and `url` from this dictionary.
*   Finally, when it calls `get_table_data`, it passes these extracted credentials to the function that uses the Azure client library (`TableServiceClient`), allowing it to authenticate the connection to Azure Table Storage.

The process is similar for other tasks that interact with Azure, such as `wir_blob_data_generator.py`, `src/create_batches.py` (which needs read access to the source blob and write access to the destination blob), and `src/image_inspection.py` (which needs read access to download sample images). They all get the keys file path from `cfg.pipeline_keys`, read the file using `read_yaml`, and use the specific keys needed for their Azure operations.

This approach ensures that your sensitive authentication keys are managed in a dedicated, separate file (`keys/authorized_keys.yaml`) and are only read by the project at runtime based on the path provided in `conf/config.yaml`, keeping them out of the main code repository for better security.

### Summary

In this final chapter, we learned about **Azure Authentication Keys**, which are sensitive credentials (like SAS tokens) required for the `Field-DataExploration` project to securely connect to and interact with your data stored in Azure Blob and Table Storage.

We understood that as a user, your role is to obtain these keys from Azure and store them in a specific YAML file, `keys/authorized_keys.yaml`, following the structure outlined in the `keys/README.md`. The location of this file is pointed to by the `pipeline_keys` setting in `conf/config.yaml`.

Under the hood, we saw that task scripts requiring Azure access receive the full configuration (`cfg`), retrieve the path to the keys file from `cfg.pipeline_keys`, use the `read_yaml` utility function ([Chapter 9: Utility Functions](09_utility_functions_.md)) to load the keys into a Python dictionary, and then use the appropriate URL and SAS token from this dictionary when initializing and using the Azure client libraries or calling tools like `azcopy`.

This method centralizes key management, separates sensitive credentials from the codebase, and allows the project to authenticate securely with Azure to perform tasks like data acquisition, batching, and image inspection.

This concludes our tutorial on the `Field-DataExploration` project! We've covered everything from how the project is configured and run, to how it acquires, processes, and reports on data, and finally, how it securely accesses the cloud storage where that data lives.

You now have a solid understanding of the key components and concepts that make this project work. We encourage you to explore the code further, experiment with the configuration settings in `conf/config.yaml`, and adapt the project to your specific data exploration needs.

Thank you for following along!

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)