import re
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from utils.utils import find_most_recent_data_csv

log = logging.getLogger(__name__)

class PreprocessingCheck:
    """
    A class to analyze image storage locations and extract information such as image counts and folder metadata.

    Attributes:
        storage_path (Path): The base path of the NFS storage to analyze.
        save_table_path (Path): Path to save the CSV table of analysis results.
        save_plot_dir (Path): Directory to save the generated plots.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes PreprocessingCheck with the base path of the NFS storage.

        Args:
            cfg (DictConfig): Configuration object containing paths for data storage and report generation.
        """
        self.storage_path = Path(cfg.paths.longterm_storage)
        if not self.storage_path.exists():
            log.error(f"Path {self.storage_path} does not exist.")
            raise FileNotFoundError(f"Path {self.storage_path} does not exist.")
        log.info(f"Initialized PreprocessingCheck for path: {self.storage_path}")
        
        # Save directory for table and plot
        self.save_table_path = Path(cfg.paths.preprocessing_analysis)  # Full path to save the table
        self.save_plot_dir = Path(cfg.paths.plots_all_years)  # Directory to save the plot

    def analyze_directory(self) -> pd.DataFrame:
        """
        Analyze the NFS storage to count image files and gather metadata for each subfolder.

        Returns:
            pd.DataFrame: A DataFrame containing folder metadata and image counts.
        """
        results = []
        # Define pattern for folder names like 'TX_2024-07-07'
        pattern = re.compile(r"^[A-Z]{2}_\d{4}-\d{2}-\d{2}$")
        
        for subdir in self.storage_path.iterdir():
            if subdir.is_dir() and pattern.match(subdir.name):
                jpg_count, raw_count = self._count_images(subdir)
                folder_metadata = self._get_folder_metadata(subdir)
                folder_data = {
                    'FolderName': subdir.name,
                    'JPGCount': jpg_count,
                    'RAWCount': raw_count,
                    'CreationDate': folder_metadata[0],
                    'LastModifiedDate': folder_metadata[1]
                }
                results.append(folder_data)
                log.info(f"Processed folder: {subdir.name} - JPG: {jpg_count}, RAW: {raw_count}")
            else:
                log.info(f"Skipped folder: {subdir.name} (does not match state abbreviation and date pattern)")
        
        # Convert results to a pandas DataFrame
        df = pd.DataFrame(results)
        log.info(f"Directory analysis completed. Processed {len(results)} folders.")
        return df

    def _count_images(self, folder: Path) -> Tuple[int, int]:
        """
        Count the number of JPG and RAW image files in a given folder.

        Args:
            folder (Path): The folder path to count images in.

        Returns:
            Tuple[int, int]: Number of JPG and RAW files in the folder.
        """
        jpg_count = sum(1 for f in folder.rglob('*.jpg'))
        raw_count = sum(1 for f in folder.rglob('*.ARW'))
        log.debug(f"Counted {jpg_count} JPG files and {raw_count} RAW files in folder: {folder}")
        return jpg_count, raw_count

    def _get_folder_metadata(self, folder: Path) -> Tuple[str, str]:
        """
        Retrieve the creation and last modified date of a folder.

        Args:
            folder (Path): The folder path to extract metadata from.

        Returns:
            Tuple[str, str]: The creation and last modified date of the folder in 'YYYY-MM-DD' format.
        """
        stat = folder.stat()
        creation_date = datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d')
        last_modified_date = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d')
        log.debug(f"Retrieved metadata for folder {folder}: CreationDate={creation_date}, LastModifiedDate={last_modified_date}")
        return creation_date, last_modified_date

    def save_to_csv(self, df: pd.DataFrame) -> None:
        """
        Save the DataFrame containing folder analysis results to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame containing folder metadata and image counts.
        """
        df.to_csv(self.save_table_path, index=False)
        log.info(f"Results saved to CSV: {self.save_table_path}")

    def plot_batches_per_week(self, df: pd.DataFrame) -> None:
        """
        Plot the number of valid batches (folders where JPGCount equals RAWCount) created per week,
        and add a text box listing batches where the counts don't match if there are any.
        
        Args:
            df (pd.DataFrame): The DataFrame containing folder metadata, including 'LastModifiedDate'.
        """
        # Create a new column 'IsEqual' to check if JPGCount equals RAWCount
        df['IsEqual'] = df['JPGCount'] == df['RAWCount']

        # Filter based on the 'IsEqual' column
        valid_folders = df.loc[df['IsEqual']].copy()
        invalid_folders = df.loc[~df['IsEqual']]  # Folders where counts don't match

        if not valid_folders.empty:
            # Convert 'LastModifiedDate' to datetime and set it as index
            valid_folders.loc[:, 'LastModifiedDate'] = pd.to_datetime(valid_folders['LastModifiedDate'])
            valid_folders.set_index('LastModifiedDate', inplace=True)
            
            valid_folders.loc[:, 'WeekStart'] = valid_folders.index.to_period('W').start_time

            # Group by week start date and count
            batches_per_week = valid_folders.groupby('WeekStart').size().reset_index(name='BatchCount')
            # Plot using Seaborn catplot
            sns.catplot(x='WeekStart', y='BatchCount', data=batches_per_week, kind='bar', height=6, aspect=2)
            plt.title('Number of Preprocessed Batches Created Per Week')
            plt.xlabel('Week Start Date')
            plt.ylabel('Number of Preprocessed Batches')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # If there are invalid folders, list them in a text box above the plot
            if not invalid_folders.empty:
                invalid_batch_list = "\n".join(invalid_folders['FolderName'].tolist())
                plt.gcf().text(0.2, 0.95, f"Batches with unequal JPG and RAW counts:\n{invalid_batch_list}",
                               ha='center', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            
            # Save the plot
            save_path = Path(self.save_plot_dir, 'preprocessed_batches_per_week.png')
            plt.savefig(save_path)
            plt.close()
            log.info(f"Plot saved to {save_path}")
        else:
            log.info("No valid folders to plot (no folders where JPGCount equals RAWCount).")

class BatchReport:
    """A class to generate reports for Field data visualization,
    focusing on different aspects of plant data.
    """

    def __init__(self, cfg) -> None:
        """Initialize the BatchReport with configuration and data loading."""
        self.cfg = cfg
        self.csv_path = find_most_recent_data_csv(cfg.paths.datadir)
        self.df = self.read()
        self.config_report_dir()
        self.config_palettes()

    def config_palettes(self) -> None:
        """Configure the color palettes for different plant types."""
        self.planttype_palette = {
            "WEEDS": "#55A868",
            "COVERCROPS": "#4C72B0",
            "CASHCROPS": "#C44E52",
        }

    def config_report_dir(self) -> None:
        """Configure and create necessary directories for report outputs."""
        self.report_dir = Path(self.cfg.paths.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)

        self.reportplot_dir = Path(self.cfg.paths.report_plots)
        self.reportplot_dir.mkdir(exist_ok=True, parents=True)

        self.plots_all_years = Path(self.cfg.paths.plots_all_years)
        self.plots_all_years.mkdir(exist_ok=True, parents=True)

    def read(self) -> pd.DataFrame:
        """Read and load data from a CSV file."""
        log.info("Reading and converting datetime columns in CSV")
        log.info(f"Reading path: {self.csv_path}")
        df = pd.read_csv(self.csv_path, dtype={"SubBatchIndex": str}, low_memory=False)

        return df

    def write_duplicate_jpgs(self, df: pd.DataFrame) -> None:
        pass

    def write_duplicate_raws(self, df: pd.DataFrame) -> None:
        pass

    def write_missing_raws(self, df: pd.DataFrame) -> None:
        """Write information on missing raw images to a CSV file."""
        # By state, species, upload_time
        columns = [
            "Name",
            "UsState",
            "PlantType",
            "Species",
            "MasterRefID",
            "BaseName",
            "Extension",
            "UploadDateUTC",
            "ImageIndex",
            "Username",
            "HasMatchingJpgAndRaw",
        ]
        df["UploadDateTimeUTC"] = pd.to_datetime(df["UploadDateTimeUTC"])
        df["UploadDateUTC"] = df["UploadDateTimeUTC"].dt.date
        df = (
            df[df["HasMatchingJpgAndRaw"] == False][columns]
            # .drop_duplicates(subset="Name")
            .reset_index(drop=True)
        )
        df.to_csv(self.cfg.paths.missing_batch_folders, index=False)
        log.info("Missing raws data written successfully.")

    def num_uploads_last_7days_by_state(self):
        """Creates a table of uploads from last 7 days by location in a csv format."""

        df = self.df.copy()
        df["UploadDateTimeUTC"] = pd.to_datetime(df["UploadDateTimeUTC"])
        df["UploadDateUTC"] = pd.to_datetime(df["UploadDateTimeUTC"].dt.date)
        current_date_time = pd.to_datetime(datetime.now().date())
        seven_days_ago = pd.to_datetime(
            current_date_time - timedelta(days=7)
        )  # calculate date 7 days ago

        df_last_7_days = df[
            (df["UploadDateUTC"] >= seven_days_ago)
            & (df["UploadDateUTC"] <= current_date_time)
        ].copy()  # filter for last 7 days
        df_last_7_days["IsDuplicated"] = df_last_7_days.duplicated("Name", keep=False)
        grouped_df_last_7_days = (
            df_last_7_days.groupby(
                [
                    "UsState",
                    "PlantType",
                    "Species",
                    "Extension",
                    "HasMatchingJpgAndRaw",
                    "IsDuplicated",
                ]
            )
            .size()
            .reset_index(name="count")
        )

        grouped_df_last_7_days.to_csv(self.cfg.paths.uploads_7days, index=False)
        log.info("Created table of uploads from last 7 days by location successfully.")

    def plot_unique_masterrefids_by_state_and_planttype(self) -> None:
        """Generate a bar plot showing the distribution of unique MasterRefIDs by state and plant type."""
        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]

        # Count the number of unique MasterRefID for each UsState and Extension
        unique_ids_count = (
            data.groupby(["UsState", "PlantType"])["MasterRefID"]
            .nunique()
            .reset_index()
        )

        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))

            bar_plot = sns.barplot(
                data=unique_ids_count,
                x="UsState",
                y="MasterRefID",
                hue="PlantType",
                palette=self.planttype_palette,
                ax=ax,
            )

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title("Unique MasterRefIDs (samples) by State and Plant Type")
            # Adding a note under the title
            ax.text(
                0.0725,
                -0.125,
                "$^{*}$HasMatchingJpgAndRaw = True",
                ha="center",
                fontsize=9,
                transform=ax.transAxes,
            )
            ax.set_ylabel("# MasterRefIDs (samples)")
            ax.set_xlabel("State Location")
            ax.legend(title="Plant Type")
            # Add labels to each bar
            for bar_container in bar_plot.containers:
                ax.bar_label(bar_container, label_type="edge", padding=3, fontsize=7)

            fig.tight_layout()
            save_path = f"{self.cfg.paths.plots_all_years}/unique_masterrefids_by_state_and_planttype.png"
            fig.savefig(save_path, dpi=300)
            log.info("Unique MasterRefIDs plot saved.")

    def plot_image_vs_raws_by_species(self):
        # Count the number of unique Images for each UsState and Extension
        unique_ids_count = (
            self.df.groupby(["UsState", "Extension"])["Name"].nunique().reset_index()
        )

        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))

            sns.barplot(
                data=unique_ids_count,
                x="UsState",
                y="Name",
                hue="Extension",
                ax=ax,
            )
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title("Number of Images by State and by Image Extension")
            ax.set_ylabel("Number of Images")
            ax.set_xlabel("State Location")
            ax.legend(title="Image Type")
            fig.tight_layout()
            save_path = (
                f"{self.cfg.paths.plots_all_years}/image_vs_raws_by_species.png"
            )
            fig.savefig(save_path, dpi=300)
            log.info("Jpg vs Raws plot saved.")

    def plot_sample_species_distribution(self):
        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]
        data = data[data["Extension"] == "arw"]

        samplecount_df = (
            data.groupby(["PlantType", "Species"])["MasterRefID"]
            .nunique()
            .reset_index(name="sample_count")
            .sort_values(by="sample_count")
        )

        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(8, 14))

            sns.barplot(
                samplecount_df,
                y="Species",
                x="sample_count",
                # kind="bar",
                hue="PlantType",
                # height=10,
                palette=self.planttype_palette,
                ax=ax,
            )
            ax.set_ylabel("Species")
            ax.set_xlabel("Number of Unique Samples")
            ax.text(
                -0.050,
                -0.035,
                "$^{*}$HasMatchingJpgAndRaw = True",
                ha="center",
                fontsize=9,
                transform=ax.transAxes,
            )
            # g.tight_layout()
            ax.figure.suptitle("Samples by Species and Plant Type", fontsize=18)
            # Adding the number of samples at the end of each bar
            for p in ax.patches:
                width = p.get_width()
                ax.text(
                    width + 1,
                    p.get_y() + p.get_height() / 2,
                    "{:1.0f}".format(width),
                    ha="left",
                    va="center",
                )
            fig.tight_layout()
            # plt.subplots_adjust(top=0.93)
            save_path = f"{self.cfg.paths.plots_all_years}/unique_masterrefids_by_species_and_planttype.png"
            fig.savefig(save_path, dpi=300)
            log.info("Species Distribution plot saved.")

    def plot_cumulative_samples_species_by_year(self):
        # Filter data based on criteria
        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]
        data = data[data["Extension"] == "arw"]
        data["UsState"] = data["UsState"].replace("NC01", "NC")

        # Extract year from CameraInfo_DateTime
        data['CameraInfo_DateTime'] = pd.to_datetime(data['CameraInfo_DateTime'])
        data['Year'] = data['CameraInfo_DateTime'].dt.year
        # Replace 2021 with 2022 because of camera config/reset error
        data['Year'] = data['Year'].replace(2021, 2022)
        
        

        # Count unique MasterRefID for each combination of UsState, Species, and Year
        samplecount_df = (
            data.groupby(["UsState", "Species", "Year"])["MasterRefID"]
            .nunique()
            .reset_index(name="sample_count")
            .sort_values(by="Year")
        )

        # Sort data by Year and calculate cumulative counts for each species
        samplecount_df["cumulative_count"] = samplecount_df.groupby(["UsState", "Species"])["sample_count"].cumsum()

        # Pivot the data for stacking bars by year
        samplecount_pivot = samplecount_df.pivot_table(
            index=["Species", "UsState"], columns="Year", values="sample_count", aggfunc="sum", fill_value=0
        )
        # Sort species names alphabetically
        samplecount_pivot = samplecount_pivot.sort_index(level="Species")

        # Plot cumulative counts as stacked bar plots for each UsState
        unique_states = samplecount_df["UsState"].unique()

        for state in unique_states:
            fig, ax = plt.subplots(figsize=(10, 7))
            state_data = samplecount_pivot.xs(state, level="UsState")
            state_data = state_data.sort_index(level="Species")

            # Plot stacked bars by year for each species
            state_data.plot(
                kind="barh", stacked=True, ax=ax, cmap="tab20"
            )
            

            # Set labels and title
            ax.set_xlabel("Cumulative Number of Unique Samples")
            ax.set_ylabel("Species")
            ax.set_title(f"Cumulative Samples by Species and Year: {state}", fontsize=18)

            # Add legend and layout adjustments
            ax.legend(title="Year", bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()

            # Save plot
            save_path = f"{self.cfg.paths.plots_all_years}/cumulative_stacked_samples_by_species_for_{state}.png"
            fig.savefig(save_path, dpi=300)
            log.info(f"Cumulative Samples by Species plot saved for {state}.")
            plt.close()


    def plot_sample_species_state_distribution(self):
        data = self.df[self.df["HasMatchingJpgAndRaw"]==True]
        data = data[data["Extension"] == "arw"]
        data["UsState"] = data["UsState"].replace("NC01", "NC")

        samplecount_df = (
            data.groupby(["UsState", "PlantType", "Species"])["MasterRefID"]
            .nunique()
            .reset_index(name="sample_count")
            .sort_values(by="sample_count")
        )

        # Create a separate bar plot for each UsState
        unique_states = samplecount_df["UsState"].unique()

        for state in unique_states:
            fig, ax = plt.subplots(figsize=(8, 6))
            state_data = samplecount_df[samplecount_df["UsState"] == state]
            sns.barplot(
                data=state_data,
                x="sample_count",
                y="Species",
                hue="PlantType",
                palette=self.planttype_palette,
                ax=ax,
            )

            # Adding the number of samples at the end of each bar
            for p in ax.patches:
                width = p.get_width()
                ax.text(
                    width + 1,
                    p.get_y() + p.get_height() / 2,
                    "{:1.0f}".format(width),
                    ha="left",
                    va="center",
                )
            ax.set_ylabel("Species")
            ax.set_xlabel("Number of Unique Samples")
            ax.text(
                -0.050,
                -0.085,
                "$^{*}$HasMatchingJpgAndRaw = True",
                ha="center",
                fontsize=9,
                transform=ax.transAxes,
            )

            ax.figure.suptitle(f"Samples by Species: {state}", fontsize=18)

            fig.tight_layout()
            save_path = f"{self.cfg.paths.plots_all_years}/unique_masterrefids_by_species_for_{state}.png"
            fig.savefig(save_path, dpi=300)
            log.info("Species Distribution plot saved.")
            plt.close()

    def plot_num_samples_season(self):
        """Generate a bar plot showing the distribution of unique MasterRefIDs by plant type."""

        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]

        # Count the number of unique MasterRefID for each Species
        unique_ids_count = (
            data.groupby(["PlantType"])["MasterRefID"]
            .nunique()
            .reset_index()
            .sort_values(by="MasterRefID")
        )

        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))

            bar_plot = sns.barplot(
                data=unique_ids_count, x="PlantType", y="MasterRefID", ax=ax, width=0.25
            )

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.figure.suptitle("Samples by Species", fontsize=18)

            # Adding a note under the title
            annot_text = "$^{*}$HasMatchingJpgAndRaw = True"
            ax.annotate(
                annot_text,
                xy=(0.05, 0.9),
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                annotation_clip=False,
            )

            ax.set_ylabel("# MasterRefIDs (samples)")
            ax.set_xlabel("Plant Type")
            # Add labels to each bar
            for bar_container in bar_plot.containers:
                ax.bar_label(bar_container, label_type="edge", padding=3, fontsize=7)

            fig.tight_layout()
            save_path = (
                f"{self.cfg.paths.plots_all_years}/unique_masterrefids_by_season.png"
            )
            fig.savefig(save_path, dpi=300)
            log.info("Unique MasterRefIDs by Plant Type plot saved.")

    def plot_num_samples_usstate(self):
        """Generate a bar plot showing the distribution of unique MasterRefIDs by UsState."""

        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]

        # Count the number of unique MasterRefID for each Species
        unique_ids_count = (
            data.groupby(["UsState"])["MasterRefID"]
            .nunique()
            .reset_index()
            .sort_values(by="MasterRefID")
        )

        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))

            bar_plot = sns.barplot(
                data=unique_ids_count, x="UsState", y="MasterRefID", ax=ax, width=0.25
            )

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.figure.suptitle("Samples by State", fontsize=18)

            # Adding a note under the title
            annot_text = "$^{*}$HasMatchingJpgAndRaw = True"
            ax.annotate(
                annot_text,
                xy=(0.05, 0.9),
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                annotation_clip=False,
            )

            ax.set_ylabel("# MasterRefIDs (samples)")
            ax.set_xlabel("State")
            # Add labels to each bar
            for bar_container in bar_plot.containers:
                ax.bar_label(bar_container, label_type="edge", padding=3, fontsize=7)

            fig.tight_layout()
            save_path = (
                f"{self.cfg.paths.plots_all_years}/unique_masterrefids_by_state.png"
            )
            fig.savefig(save_path, dpi=300)
            log.info("Unique MasterRefIDs by UsState plot saved.")


def main(cfg: DictConfig) -> None:
    """Main function to execute batch report tasks."""
    log.info(f"Starting {cfg.general.task}")
    batchrep = BatchReport(cfg)
    batchrep.write_missing_raws(batchrep.df)
    batchrep.plot_unique_masterrefids_by_state_and_planttype()
    batchrep.plot_sample_species_distribution()
    batchrep.plot_image_vs_raws_by_species()

    batchrep.plot_num_samples_season()
    batchrep.plot_num_samples_usstate()
    batchrep.plot_sample_species_state_distribution()
    batchrep.plot_cumulative_samples_species_by_year()
    batchrep.num_uploads_last_7days_by_state()
    
    # Start PreprocessingCheck
    analyzer = PreprocessingCheck(cfg)
    analysis_results_df = analyzer.analyze_directory()
    # Plot batches per week
    analyzer.plot_batches_per_week(analysis_results_df)
    # Save preprocessed batches analysis results to CSV
    analyzer.save_to_csv(analysis_results_df)
    
    log.info(f"{cfg.general.task} completed.")
