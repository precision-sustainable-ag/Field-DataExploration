# Field-DataExploration

**Description**:
This repo is the central hub for the initial phase of data exploration and assessment of a project aimed at comprehensive data management and processing of the PSA Field data. It focuses on addressing the initial backlog of images and metadata while emphasizing a deep understanding of our existing Field images. This phase is critical for laying the groundwork for advanced data processing and management in later stages of the project.

**Key Features**:

1. **Data Volume Assessment**: Contains tools and methodologies for evaluating the size and complexity of the existing data pool, ensuring an understanding of the scale of data we're managing.

2. **Data Visualization and Status Reporting**: Features scripts and resources for creating visual representations of the data's current status and contents. This aids in identifying patterns, anomalies, and key areas requiring attention.

3. **Metadata Quality Review and Image Sampling**: Offers guidelines and tools for inspecting metadata accuracy and completeness, along with methods for performing quality checks on image samples.

4. **Data Exploration and Reporting**: Includes exploratory data analysis tools to understand the characteristics and structure of the dataset. It also encompasses reporting mechanisms for documenting findings and progress.

5. **Data Organization and Issue Resolution**: Provides strategies and scripts for identifying immediate dataset issues, along with solutions to these challenges.

## Installation and Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage the Python environment and dependencies.

1. Clone this repository to your local machine.
2. Navigate to the repository directory in your terminal.
3. Run the setup script, which installs `uv` if needed, provisions Python, and syncs dependencies from `pyproject.toml`/`uv.lock`:
   ```bash
   ./setup.sh
   ```
4. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```
   Or run commands without activating it, e.g. `uv run python main.py`.

### Running the Pipeline
The pipeline is DB-driven (SQLite) and config-driven via [Hydra](conf/config.yaml).

1. Set which tasks to run under `pipeline:` in [conf/config.yaml](conf/config.yaml#L11).
2. Run the pipeline:
   ```bash
   python main.py
   ```

## Major Scripts

Pipeline tasks live under `src/` and are wired up by name in `conf/config.yaml`'s `pipeline` list.

### `wir_table_generator` / `wir_blob_data_generator`
Pull image reference, sample attribute, and blob metadata from the configured Azure sources (see `conf/config.yaml`'s `sources`) and upsert them into the SQLite datastore.

### `merge_samples`
Coalesces per-source sample attributes and locations into merged records in the datastore.

### `append_datetime_db`
Downloads each image's JPG from Azure Blob just long enough to read its EXIF capture DateTime, then discards the file and records the timestamp in the DB. This must run before `create_batches_db`; images already have a datetime are skipped. Uses `ThreadPoolExecutor` for concurrent downloads/reads.

### `create_batches_db`
Groups images into "batches" using the DateTime information from `append_datetime_db`, based on State, capture date, and 3-hour capture-time intervals, then copies each batch to the `field-batches` blob container. Skips batches already processed. Can run concurrently or sequentially.

### `report_db` / `plot_by_season_db`
Generate status reports and plots (by location, by season) from the current DB state. Output is written under `report/<date>/`.

### `migrate_to_db`
One-off migration of legacy CSV-based batch/location data into the SQLite datastore.

### `image_inspection_db`
Facilitates manual quality checks: randomly selects recently-uploaded images and plots each alongside its key metadata fields. Plots are located in the `report/<date>/inspection` folder.