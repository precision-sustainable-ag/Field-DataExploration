# Chapter 4: Data Integration & Preprocessing

Welcome back! In [Chapter 1: Configuration Management](01_configuration_management_.md), we learned how our project gets its settings. In [Chapter 2: Pipeline Task Runner](02_pipeline_task_runner_.md), we saw how these settings are used to run different parts (tasks) of the project automatically. Most recently, in [Chapter 3: Azure Data Acquisition](03_azure_data_acquisition_.md), we learned how the project connects to Azure to download the raw lists of files (Blob metrics) and metadata records (Table data) and saves them as CSV files locally.

Now that we have these raw pieces of data saved on our computer, they aren't immediately ready for analysis. They are separate files, might have slightly different formats, or might contain information we don't need yet. This is where **Data Integration & Preprocessing** comes in!

### What is Data Integration & Preprocessing?

Imagine you've just received several boxes of documents related to an important project. One box has a list of all the files, another has notes about some of those files, and maybe a third has corrections or updates. To make sense of everything, you wouldn't keep them in separate boxes. You'd:

1.  **Combine:** Take papers from different boxes that belong together (e.g., a file description from one box and notes about that specific file from another) and put them side-by-side.
2.  **Clean:** Remove any duplicate papers, fix typos, or discard papers that aren't relevant.
3.  **Structure:** Organize the combined papers into a single, neat binder or master file, making sure similar information is always in the same place (like having a dedicated column for "File Name" or "Notes").

**Data Integration & Preprocessing** in our project is exactly this process for the digital data we acquired from Azure. It's the step where the raw lists of files (Blob metrics) and the various metadata records (from different Azure Tables) are:

*   **Combined:** Merged together based on common information (like an image name or a unique identifier).
*   **Cleaned:** Handle missing values, maybe remove unnecessary columns, and standardize formats.
*   **Structured:** Organized into one single, comprehensive dataset (a big table) that's easy for the next steps in the pipeline to understand and analyze.

**The Central Use Case:** The main goal of this step is to take the raw CSV files downloaded during [Azure Data Acquisition](03_azure_data_acquisition_.md) (like `weedsimagerepo_blob_metrics.csv` and several `*_table_metrics.csv` files) and produce a single, clean CSV file (often named `merged_blobs_tables_metadata.csv`) that contains *all* relevant information combined for each image or record. This unified dataset is the foundation for all subsequent analysis and reporting.

### How to Use Data Integration & Preprocessing

As with other steps, you don't directly run the data integration code. You tell the [Pipeline Task Runner](02_pipeline_task_runner_.md) to execute the necessary tasks by including their names in the `pipeline` list in your `conf/config.yaml` file.

The two primary tasks involved in this step are:

*   `process_blob_analysis`: This task takes the Blob metrics CSV and the `wirimagerefs_table_metrics.csv` (which links image names to MasterRefIDs) and performs an initial merge.
*   `process_tables_analysis`: This task takes the output of `process_blob_analysis` and merges it with data from the other Azure Table CSVs (like `wirmastermeta_table_metrics.csv`, `wircovercropsmeta_table_metrics.csv`, etc.), performs further cleaning, renaming, and structuring.

To include data integration in your project run, make sure these tasks are listed in your `pipeline` *after* the data acquisition tasks (`wir_table_generator`, `wir_blob_data_generator`) because they need the data files that the acquisition tasks create.

Here's an example of how your `conf/config.yaml` might look to acquire data and then immediately process it:

```yaml
# conf/config.yaml (Snippet showing Data Integration tasks)

# ... other settings ...

pipeline:
    - wir_table_generator     # Acquire Table data (from Ch 3)
    - wir_blob_data_generator # Acquire Blob metrics (from Ch 3)
    - process_blob_analysis   # Merge Blob metrics and image references
    - process_tables_analysis # Merge with other tables and clean/structure
    # - append_datetime       # (Optional) Next step in pipeline...

# ... other settings ...
```

When you run `python main.py` with this configuration, the [Pipeline Task Runner](02_pipeline_task_runner_.md) will execute these tasks in the specified order. The `process_blob_analysis` task will run, use the CSVs saved by the acquisition tasks, and save a temporary merged CSV. Then, the `process_tables_analysis` task will run, read that temporary CSV and the other acquired Table CSVs, perform more merging and cleaning, and save the final, unified `merged_blobs_tables_metadata.csv`.

### How It Works Under the Hood

Let's trace the journey of the data through the integration and preprocessing steps:

```mermaid
sequenceDiagram
    participant User
    participant MainPy as main.py (Task Runner)
    participant ProcessBlob as process_blob_analysis.py
    participant ProcessTables as process_tables_analysis.py
    participant AcquiredCSVs as Local Raw CSVs (from Ch 3)
    participant PandasLib as Pandas Library
    participant MergedCSV as merged_blobs_tables_metadata.csv

    User->MainPy: Run script (python main.py)
    MainPy->MainPy: Get task list from cfg.pipeline
    loop For process_blob_analysis
        MainPy->ProcessBlob: Run main(cfg)
        ProcessBlob->AcquiredCSVs: Read blob_metrics.csv & wirimagerefs_table_metrics.csv
        ProcessBlob->PandasLib: Use Pandas to merge DataFrames
        ProcessBlob->AcquiredCSVs: Save temp merged CSV
        AcquiredCSVs-->ProcessBlob: Confirm save
    end
    loop For process_tables_analysis
        MainPy->ProcessTables: Run main(cfg)
        ProcessTables->AcquiredCSVs: Read temp merged CSV & other *_table_metrics.csv files
        ProcessTables->PandasLib: Use Pandas to merge and clean DataFrames
        ProcessTables->MergedCSV: Save final merged CSV
        MergedCSV-->ProcessTables: Confirm save
    end
    MainPy: Integration tasks completed!
```

This diagram shows that the Task Runner triggers the two processing scripts. Each script reads necessary input files (CSVs saved in the previous step), uses the powerful `pandas` library to do the merging, cleaning, and structuring, and finally saves the resulting data into new CSV files.

Let's look at simplified snippets from the actual files:

#### Initial Merge (`src/process_blob_analysis.py`)

This file is responsible for the first merging step. It takes the raw blob metrics (list of files from Azure Blob Storage) and combines it with the `wirimagerefs` table data (which contains links between image names and MasterRefIDs from Azure Table Storage).

```python
# src/process_blob_analysis.py (Simplified Snippet)
import os # For joining paths
from omegaconf import DictConfig
import pandas as pd # The data processing powerhouse!
import logging
from utils.utils import read_csv_as_df # Helper function

log = logging.getLogger(__name__)

class BlobTablePreProcessing:
    def __init__(self, cfg: DictConfig) -> None:
        # Get the paths to input & output directories from config
        self.blob_table_dir = cfg.paths.blobsdir # Where blob metrics CSV is
        self.refs_table_dir = cfg.paths.tablesdir # Where wirimagerefs CSV is
        self.processed_datadir = cfg.paths.processed_datadir # Where to save outputs

        # Read the input CSVs using the helper function
        blob_fname = "weedsimagerepo_blob_metrics.csv"
        table_fname = "wirimagerefs_table_metrics.csv"
        self.blobs_csv = read_csv_as_df(os.path.join(self.blob_table_dir, blob_fname))
        self.imagerefs_csv = read_csv_as_df(os.path.join(self.refs_table_dir, table_fname))

        # Call the main processing method
        self.preprocess_imgrefs(self.blobs_csv, self.imagerefs_csv)

    def preprocess_imgrefs(self, blobs_df, imageref_df):
        # Extract just the filename from the ImageURL in imageref_df
        imageref_df["name"] = imageref_df["ImageURL"].apply(lambda url: os.path.basename(url))

        # Merge the two dataframes using the 'name' column
        # Left join keeps all rows from the blob_metrics (blobs_df)
        processed_blobs = pd.merge(blobs_df, imageref_df, on="name", how="left")

        # Identify rows from blobs_df that didn't have a match in imageref_df
        missing_rows = processed_blobs[processed_blobs['MasterRefID'].isna()]

        # Keep only the rows that *did* have a match (have a MasterRefID)
        processed_blobs = processed_blobs[processed_blobs['MasterRefID'].notna()]

        # --- More complex cleaning/renaming happens here ---
        # (Simplified for brevity)

        # Save the results to CSV files
        processed_path = Path(self.processed_datadir, 'merged_blobs_tables_metadata.csv')
        missing_path = Path(self.processed_datadir, 'missing_blobs_metadata.csv')
        processed_blobs.to_csv(processed_path, index=False)
        missing_rows.to_csv(missing_path, index=False)
        log.info(f"Saved merged blobs to {processed_path}")

def main(cfg: DictConfig) -> None:
    # This is the entry point called by the Task Runner
    log.info("Starting blob analysis preprocessing")
    BlobTablePreProcessing(cfg) # Create an instance, passing config
    log.info("Blob analysis preprocessing completed.")
```

**Explanation:**

*   The `main(cfg: DictConfig)` function is the entry point, receiving `cfg`.
*   It creates a `BlobTablePreProcessing` object, passing `cfg`.
*   The `__init__` method inside the class uses `cfg.paths` to find the locations of the input CSV files (saved from [Chapter 3: Azure Data Acquisition](03_azure_data_acquisition_.md)) and where to save outputs. It reads these CSVs into pandas DataFrames using the `read_csv_as_df` helper (from `utils.utils.py`).
*   The `preprocess_imgrefs` method then performs the core logic: it extracts the filename from the `ImageURL` column in one DataFrame, and then uses `pd.merge` to combine it with the other DataFrame based on this filename (`on="name"`). It performs a "left" merge, which means it keeps all rows from the `blobs_csv` and adds matching information from `imagerefs_csv` where available.
*   It then separates the successfully merged data from the rows that didn't find a match (`missing_rows`).
*   Finally, it saves both the successfully merged data and the list of missing rows into separate CSV files in the directory specified by `cfg.paths.processed_datadir`. The main output `merged_blobs_tables_metadata.csv` now contains blob metrics combined with the `MasterRefID` and other details from `wirimagerefs`.

#### Merging with More Tables and Final Structuring (`src/process_tables_analysis.py`)

This file takes the output CSV from `process_blob_analysis` (`merged_blobs_tables_metadata.csv`) and merges it with the data from the *other* Azure Table CSVs (like `wirmastermeta`, `wircovercropsmeta`, `wircropsmeta`, `wirweedsmeta`). It also performs significant cleaning, renaming, and restructuring to create the final, comprehensive dataset.

```python
# src/process_tables_analysis.py (Simplified Snippet)
import os
from omegaconf import DictConfig
import pandas as pd
import logging
from utils.utils import read_csv_as_df # Helper function

log = logging.getLogger(__name__)

class WIRTablesPreProcessing:
    def __init__(self, cfg: DictConfig) -> None:
        # Get the paths to input & output directories from config
        self.tables_dir = cfg.paths.tablesdir # Where other table CSVs are
        self.processed_datadir = cfg.paths.processed_datadir # Where merged_blobs_tables_metadata.csv is and output goes

        # Read the required input CSVs
        self.wirmastermeta_df = read_csv_as_df(os.path.join(self.tables_dir, "wirmastermeta_table_metrics.csv"))
        self.wircovercropsmeta_df = read_csv_as_df(os.path.join(self.tables_dir, "wircovercropsmeta_table_metrics.csv"))
        # ... read other table CSVs ... (Simplified)
        self.wirmergedtable_df = read_csv_as_df(os.path.join(self.processed_datadir, "merged_blobs_tables_metadata.csv")) # Output from process_blob_analysis!

        # Perform the main processing
        self.process_wir_tables()

    def process_wir_tables(self):
        # --- Start merging with other tables ---

        # Merge the base data (from process_blob_analysis) with MasterMeta
        # Merging typically happens on common columns like 'MasterRefID'
        processed_table = pd.merge(self.wirmergedtable_df, self.wirmastermeta_df, how="left", on=["MasterRefID", "PartitionKey"])

        # Merge the result with CoverCrops data
        processed_table = pd.merge(processed_table, self.wircovercropsmeta_df, how="left", on=["MasterRefID", "CloudCover", "GroundResidue", "GroundCover"])

        # ... merge with Crops, Weeds tables similarly ... (Simplified)

        # --- Data Cleaning and Structuring ---

        # Example: Create a 'Species' column by combining values from different table columns
        processed_table["Species"] = processed_table["WeedType"].fillna(processed_table["CoverCropSpecies"])
        # ... continue filling 'Species' from other columns ... (Simplified)

        # Example: Rename columns for clarity
        processed_table.rename(columns={"creation_time_utc": "UploadDateTimeUTC"}, inplace=True)
        processed_table.rename(columns={"memory_mb": "SizeMiB"}, inplace=True)

        # Example: Drop redundant or intermediate columns
        processed_table = processed_table.drop('WeedType', axis=1)
        # ... drop other unnecessary columns ... (Simplified)

        # Example: Standardize values (like SizeClass)
        processed_table["SizeClass"] = processed_table["SizeClass"].replace({'Large': 'LARGE', 'Medium': 'MEDIUM', 'Small':'SMALL'})

        # --- Reorder columns to a standard format ---
        # (Simplified - the actual list is quite long)
        final_cols = ["Name", "UploadDateTimeUTC", "MasterRefID", "Species", "SizeClass", ...] # etc.
        processed_table = processed_table[final_cols]

        # Save the final, integrated, and preprocessed data
        csv_path = Path(self.processed_datadir, "merged_blobs_tables_metadata.csv") # Overwrites the temp file
        processed_table.to_csv(csv_path, index=False)
        log.info(f"Saved final processed tables data to {csv_path}")

def main(cfg: DictConfig) -> None:
    # This is the entry point called by the Task Runner
    log.info("Starting table analysis preprocessing")
    WIRTablesPreProcessing(cfg) # Create an instance, passing config
    log.info("Table analysis preprocessing completed.")
```

**Explanation:**

*   Similar to the previous file, `main(cfg: DictConfig)` is the entry point, creating a `WIRTablesPreProcessing` object.
*   The `__init__` method reads the *output* CSV from `process_blob_analysis` (`merged_blobs_tables_metadata.csv`) and the remaining raw table CSVs (from [Chapter 3: Azure Data Acquisition](03_azure_data_acquisition_.md)) into pandas DataFrames.
*   The `process_wir_tables` method is where the bulk of the work happens. It performs multiple `pd.merge` operations, progressively combining the data from different tables onto the base data, using common columns like `MasterRefID`.
*   After merging, it performs various cleaning and structuring steps:
    *   Combining information from related columns into new, standardized columns (like creating a single `Species` column from multiple potential source columns).
    *   Renaming columns to be more descriptive and consistent (`creation_time_utc` becomes `UploadDateTimeUTC`).
    *   Dropping columns that are no longer needed or were just temporary during merging.
    *   Standardizing values within columns (e.g., ensuring `SizeClass` is always uppercase).
*   Finally, it reorders the columns to a predefined logical order and saves the resulting DataFrame to the final `merged_blobs_tables_metadata.csv` file in the directory specified by `cfg.paths.processed_datadir`, overwriting the temporary file created by `process_blob_analysis`.

After these two tasks (`process_blob_analysis` and `process_tables_analysis`) have run successfully, you will have a single, clean, and well-structured CSV file (`merged_blobs_tables_metadata.csv`) containing the integrated data from all the relevant Azure sources. This file is the primary input for all subsequent analysis and reporting steps in the pipeline.

### Summary

In this chapter, we explored **Data Integration & Preprocessing**. We learned that this crucial step takes the raw data acquired from different Azure sources (Blob metrics and various Table records) and combines, cleans, and structures it into a single, unified dataset.

We saw that this is achieved by including the `process_blob_analysis` and `process_tables_analysis` tasks in the `pipeline` list in `conf/config.yaml`, after the data acquisition tasks.

Under the hood, these tasks are implemented in `src/process_blob_analysis.py` and `src/process_tables_analysis.py`. They use the `pandas` library to read the raw CSV files saved in the previous step, perform merging based on common identifiers, clean the data by handling missing values, renaming columns, and standardizing formats, and finally save the result as a single, comprehensive CSV file (`merged_blobs_tables_metadata.csv`) that is ready for further analysis.

Now that we have our clean, integrated dataset, we can start adding more valuable information to it that wasn't directly available in the raw source data.

Let's move on to the next chapter where we'll learn about **Datetime Metadata Enrichment**.

[Datetime Metadata Enrichment](05_datetime_metadata_enrichment_.md)

---

Generated by [AI Codebase Knowledge Builder](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)