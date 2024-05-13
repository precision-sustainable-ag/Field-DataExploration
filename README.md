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

### Installing Conda
To manage the project's dependencies efficiently, we use Conda, a powerful package manager and environment manager. Follow these steps to install Conda if you haven't already:

1. Download the appropriate version of Miniconda for your operating system from the official [Miniconda website](https://docs.anaconda.com/free/miniconda/).
2. Follow the installation instructions provided on the website for your OS. This typically involves running the installer from the command line and following the on-screen prompts.
3. Once installed, open a new terminal window and type `conda list` to ensure Conda was installed correctly. You should see a list of installed packages.


### Setting Up Your Environment Using an Environment File
After installing Conda, you can set up an environment for this project using an environment file, which specifies all necessary dependencies. Here's how:

1. Clone this repository to your local machine.
2. Navigate to the repository directory in your terminal.
3. Locate the `environment.yaml` file in the repository. This file contains the list of packages needed for the project.
4. Create a new Conda environment by running the following command:
   ```bash
   conda env create -f environment.yaml
   ```
   This command reads the `environment.yaml` file and creates an environment with the name and dependencies specified within it.

5. Once the environment is created, activate it with:
   ```bash
   conda activate <env_name>
   ```
   Replace `<env_name>` with the name of the environment specified in the `environment.yaml` file.


### Running the Script
With the environment set up and activated, you can run the scripts provided in the repository to begin data exploration and analysis:

1. Ensure your Conda environment is activated:
   ```
   conda activate field
   ```
2. To run a script, use the following command syntax:
   ```bash
   sh run_volume_assessment.sh
   ```
3. [NOTE] Setup the pipeline in the main [config](conf/config.yaml#L11). To run a script, use the following command syntax:
   ```bash
   python FIELD_REPORT.py
   ```

## Major Scripts

### `append_datetime`
This script main purpose is to quickley get image DateTime information that can later be used to form batches. We quickley read jpg exif data to get capture datetime information, then remove the download jpg. We get the jpg from the azure blob container. This script appends date-time metadata extracted from the downloaded image EXIF data to an existing persistent CSV table. this is a crucial step that allows us to form "batches" for preprocessing. The script must be run before the "create_batches" task. If no new data is found, nothing happens. 

Features of Note:
   - The script finds the most recent "merged..." CSV file in "data/processed_data" and merges it with persistent data for continuous updates.
   - It processes each JPG image to extract and append EXIF DateTime metadata to the relevant records in the CSV.
   - Concurrency Handling: We use concurrent.futures and ThreadPoolExecutor for handling multiple downloads and reading of exif data at the same time.

### `create_bathces`
This scripts create batches by using the updated DateTime information from `append_datetime` , organizes raw images into "batches", and copies those image batches to the field-batches blob container. The script adjusts and groups images based on metadata into "batches" and filters out already processed or duplicate batches. Batch groupings are based on State, capture date, and 3 hour capture time intervals. It offers the flexibility to process data either concurrently or sequentially.
### `image_inspection`

This script is designed to facilitate the quality check process by performing the following functions:

Random Image Selection: Automatically selects up to 15 images that have been uploaded in the past 15 days from a merged data table.
Image Plotting with Metadata: For each selected image, the script generates a plot that includes the image itself along with key metadata fields. Plots are located in the `report/<date>/inspection` folder.