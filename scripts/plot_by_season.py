import logging
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from omegaconf import DictConfig

from utils.utils import find_most_recent_data_csv

log = logging.getLogger(__name__)

class PlotsBySeason:
    """
    A class to generate plots and tables of unique samples and plant type by season.
    """
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize the PlotsBySeason class with configuration and data loading.

        Args:
            cfg (DictConfig): Configuration object with necessary settings.
        """
        log.debug("Initializing PlotsBySeason class.")
        self.cfg = cfg
        self.state_list = cfg.state_list
        self.csv_path = find_most_recent_data_csv(cfg.data.datadir)
        self.config_report_dir()
        self.permanent_df = pd.read_csv(self.cfg.data.permanent_merged_table, low_memory=False)
        self.current_year = datetime.now().year

        # Convert date strings to datetime objects
        try:
            self.permanent_df["CameraInfo_DateTime"] = pd.to_datetime(self.permanent_df["CameraInfo_DateTime"], format="%Y-%m-%d %H:%M:%S")
        except Exception as e:
            log.warning(f"Error occurred while converting CameraInfo_DateTime: {e}. Dropping rows with invalid dates.")
            self.permanent_df["CameraInfo_DateTime"] = pd.to_datetime(self.permanent_df["CameraInfo_DateTime"], errors='coerce', format="%Y-%m-%d %H:%M:%S")
            self.permanent_df = self.permanent_df.dropna(subset=["CameraInfo_DateTime"])  # Drop rows with invalid dates

    def config_report_dir(self) -> None:
        """
        Configure and create necessary directories for report outputs.
        """
        log.debug("Configuring report directories.")
        self.report_dir = Path(self.cfg.report.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)

        self.reportplot_dir = Path(self.cfg.report.report_plots)
        self.reportplot_dir.mkdir(exist_ok=True, parents=True)

        self.plots_current_season = Path(self.cfg.report.plots_current_season)
        self.plots_current_season.mkdir(exist_ok=True, parents=True)

    def add_season_column(self) -> pd.DataFrame:
        """
        Add a "Season" column to the data DataFrame based on the CameraInfo_DateTime and PlantType columns.

        Returns:
            pd.DataFrame: The updated DataFrame with the "Season" column added.
        """
        log.info("Adding 'Season' column to the data.")
        self.permanent_df["Season"] = " "
        for index, row in self.permanent_df.iterrows():
            try:
                plant_type = row["PlantType"]
                date_time = row["CameraInfo_DateTime"]
                if plant_type in ["WEEDS", "CASHCROPS"]:
                    self.permanent_df.at[index, "Season"] = f"{date_time.year} {plant_type}"
                else:
                    if date_time >= pd.Timestamp(year=date_time.year, month=10, day=1):
                        self.permanent_df.at[index, "Season"] = f"{date_time.year}/{date_time.year + 1} {plant_type}"
                    else:
                        self.permanent_df.at[index, "Season"] = f"{date_time.year - 1}/{date_time.year} {plant_type}"
            except Exception as e:
                log.warning(f"Error processing row {index}: {e}")
                self.permanent_df.at[index, "Season"] = np.nan

        current_season_cover_crop = f"{self.current_year - 1}/{self.current_year} COVERCROPS"
        current_season_weeds = f"{self.current_year} WEEDS"
        current_season_cashcrops = f"{self.current_year} CASHCROPS"

        current_seasons = [current_season_cover_crop, current_season_weeds, current_season_cashcrops]
        data_current_season = self.permanent_df[self.permanent_df["Season"].isin(current_seasons)]

        log.info("Season column added successfully.")
        return data_current_season
        
    def plot_unique_samples_state_plant_current_season(self, data_current_season) -> None:
        """
        Generate a bar plot showing the distribution of unique MasterRefIDs by state and plant type in the current season.
        """
        log.info("Generating bar plot for unique samples by state and plant type for the current season.")

        # DataFrame filtered for the rows with both jpg and raw
        data = data_current_season[data_current_season["HasMatchingJpgAndRaw"] == True].copy() 

        # Define current season labels
        last_year = self.current_year - 1
        cover_crop_label = f"{last_year}/{self.current_year} COVERCROPS"
        weeds_label = f"{self.current_year} WEEDS"
        cash_crops_label = f"{self.current_year} CASHCROPS"
        
        # Define plant type palette with new labels
        planttype_palette = {
            cover_crop_label: "#4C72B0",
            weeds_label: "#55A868",
            cash_crops_label: "#C44E52",
        }
        
        unique_ids_count = (
            data.groupby(["UsState", "Season"])["MasterRefID"]
            .nunique()
            .reset_index()
        )
        
        # Ensure all states are included in the plot
        existing_states = set(unique_ids_count["UsState"])
        missing_states = [state for state in self.state_list if state not in existing_states]
        missing_states_df = pd.DataFrame({"UsState": missing_states})
        unique_ids_count_all_states = pd.concat([unique_ids_count, missing_states_df], ignore_index=True).sort_values(by="UsState")
        
        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bar_plot = sns.barplot(
                data=unique_ids_count_all_states,
                x="UsState",
                y="MasterRefID",
                hue="Season",
                palette=planttype_palette,
                hue_order=[cover_crop_label, weeds_label, cash_crops_label],
                ax=ax,
            )
            
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title(f"{self.current_year} Unique MasterRefIDs (samples) by State and Plant Type")
            ax.text(.0725, -.125, "$^{*}$HasMatchingJpgAndRaw = True", ha='center', fontsize=9, transform=ax.transAxes)
            ax.set_ylabel("# MasterRefIDs (samples)")
            ax.set_xlabel("State Location")
            ax.legend(title="Plant Type")
            
            # Add labels to each bar
            for bar_container in bar_plot.containers:
                ax.bar_label(bar_container, label_type='edge', padding=3, fontsize=7)
            
            fig.tight_layout()
            save_path = f"{self.cfg.report.plots_current_season}/unique_masterrefids_by_state_and_planttype_current_season.png"
            fig.savefig(save_path, dpi=300)
        
        log.info("Unique MasterRefIDs by state and plant type for current season plot saved.")

    def plot_unique_samples_species_current_season(self, data_current_season) -> None:
        """
        Generate a bar plot showing the distribution of unique MasterRefIDs by species in the current season.
        """
        log.info("Generating bar plot for unique samples by species for the current season.")

        # DataFrame filtered for the rows with both jpg and raw
        data = data_current_season[data_current_season["HasMatchingJpgAndRaw"] == True].copy() 

        unique_ids_count = (
            data.groupby(["Species"])["MasterRefID"]
            .nunique()
            .reset_index(name="sample_count")
            .sort_values(by="sample_count")
        )

        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 8))

            x_positions = range(len(unique_ids_count))
            bars = ax.bar(
                x=x_positions,
                height=unique_ids_count["sample_count"],
                color="#C44E52",
                edgecolor="black",
                width= 0.5
            )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(unique_ids_count["Species"], rotation=45)
            ax.set_ylabel("Number of Unique Samples")
            ax.set_xlabel("Species")
            ax.text(0.5, -0.25, "$^{*}$HasMatchingJpgAndRaw = True", ha='center', fontsize=9, transform=ax.transAxes)
            ax.set_title(f"{self.current_year} Samples by Species")

            # Adding the number of samples on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.0f}",
                    ha="center",
                    va="bottom"
                )

            fig.tight_layout()
            save_path = f"{self.cfg.report.plots_current_season}/unique_masterrefids_by_species_current_season.png"
            fig.savefig(save_path, dpi=300)
        log.info("Species distribution for current season plot saved.")

    def plot_image_vs_raws_by_species_current_season(self, data_current_season):
        # Count the number of unique Images for each UsState and Extension
        unique_ids_count = (
            data_current_season.groupby(["UsState", "Extension"])["Name"]
            .nunique()
            .reset_index()
        )

         # Ensure all states are included in the plot
         
        existing_states = set(unique_ids_count["UsState"])
        missing_states = [state for state in self.state_list if state not in existing_states]
        
        missing_states_df = pd.DataFrame({"UsState": missing_states})
        unique_ids_count_all_states = pd.concat([unique_ids_count, missing_states_df], ignore_index=True).sort_values(by="UsState")

        # Replace NaN values in 'Name' column with 0
        unique_ids_count_all_states['Name'] = unique_ids_count_all_states['Name'].fillna(0)
        # Replace NaN values in 'Extension' column with 'None'
        unique_ids_count_all_states['Extension'] = unique_ids_count_all_states['Extension'].fillna('None')
        # Duplicate rows with 'None' in 'Extension' column for both 'arw' and 'jpg' extensions with 0 values
        df_with_duplicates = unique_ids_count_all_states[unique_ids_count_all_states['Extension'] == 'None'].copy()
        df_with_duplicates['Extension'] = 'arw'
        df_with_duplicates_jpg = df_with_duplicates.copy()
        df_with_duplicates_jpg['Extension'] = 'jpg'

        # Concatenate the original DataFrame with the duplicated rows
        df_extended = pd.concat([unique_ids_count_all_states, df_with_duplicates, df_with_duplicates_jpg], ignore_index=True)

        # Remove original 'None' extension rows
        df_extended = df_extended[df_extended['Extension'] != 'None']

        # Remove entries with 'NC01' in 'UsState' column
        df_cleaned = df_extended[~df_extended['UsState'].isin(['NC01'])]

        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))

            sns.barplot(
                data=df_cleaned,
                x="UsState",
                y="Name",
                hue="Extension",
                ax=ax,
            )
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title(
                f"{self.current_year} Number of Images by State and by Image Extension"
            )
            ax.set_ylabel("Number of Images")
            ax.set_xlabel("State Location")
            ax.legend(title="Image Type")
            fig.tight_layout()
            save_path = (
                f"{self.cfg.report.plots_current_season}/image_vs_raws_by_species_current_season.png"
            )
            fig.savefig(save_path, dpi=300)
            log.info("Jpg vs Raws plot saved for current season.")

def main(cfg: DictConfig) -> None:
    """
    Main function to execute batch report tasks.

    Args:
        cfg (DictConfig): Configuration object with necessary settings.
    """
    log.info(f"Starting task: {cfg.general.task}")
    plots_season = PlotsBySeason(cfg)
    data_current_season = plots_season.add_season_column()
    plots_season.plot_unique_samples_state_plant_current_season(data_current_season )
    plots_season.plot_unique_samples_species_current_season(data_current_season )
    plots_season.plot_image_vs_raws_by_species_current_season(data_current_season )
    log.info(f"Task {cfg.general.task} completed.")