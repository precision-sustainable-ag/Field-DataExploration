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
        self.csv_path = find_most_recent_data_csv(cfg.data.datadir)
        self.config_report_dir()
        self.config_palettes()
        self.permanent_df = pd.read_csv(self.cfg.data.permanent_merged_table, low_memory=False)
        self.data = self.permanent_df[self.permanent_df["HasMatchingJpgAndRaw"] == True].copy()

        # Convert date strings to datetime objects
        try:
            self.data["CameraInfo_DateTime"] = pd.to_datetime(self.data["CameraInfo_DateTime"], format="%Y-%m-%d %H:%M:%S")
        except Exception as e:
            log.warning(f"Error occurred while converting CameraInfo_DateTime: {e}. Dropping rows with invalid dates.")
            self.data["CameraInfo_DateTime"] = pd.to_datetime(self.data["CameraInfo_DateTime"], errors='coerce', format="%Y-%m-%d %H:%M:%S")
            self.data = self.data.dropna(subset=["CameraInfo_DateTime"])  # Drop rows with invalid dates

    def config_palettes(self) -> None:
        """
        Configure the color palettes for different plant types.
        """
        log.debug("Configuring color palettes for plant types.")
        self.planttype_palette = {
            "WEEDS": "#55A868",
            "COVERCROPS": "#4C72B0",
            "CASHCROPS": "#C44E52",
        }

    def config_report_dir(self) -> None:
        """
        Configure and create necessary directories for report outputs.
        """
        log.debug("Configuring report directories.")
        self.report_dir = Path(self.cfg.report.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)

        self.reportplot_dir = Path(self.cfg.report.report_plots)
        self.reportplot_dir.mkdir(exist_ok=True, parents=True)

    def add_season_column(self) -> pd.DataFrame:
        """
        Add a "Season" column to the data DataFrame based on the CameraInfo_DateTime and PlantType columns.

        Returns:
            pd.DataFrame: The updated DataFrame with the "Season" column added.
        """
        log.info("Adding 'Season' column to the data.")
        self.data["Season"] = " "
        for index, row in self.data.iterrows():
            try:
                plant_type = row["PlantType"]
                date_time = row["CameraInfo_DateTime"]
                if plant_type in ["WEEDS", "CASHCROPS"]:
                    self.data.at[index, "Season"] = f"{date_time.year} {plant_type}"
                else:
                    if date_time >= pd.Timestamp(year=date_time.year, month=10, day=1):
                        self.data.at[index, "Season"] = f"{date_time.year}/{date_time.year + 1} {plant_type}"
                    else:
                        self.data.at[index, "Season"] = f"{date_time.year - 1}/{date_time.year} {plant_type}"
            except Exception as e:
                log.warning(f"Error processing row {index}: {e}")
                self.data.at[index, "Season"] = np.nan

        self.current_year = datetime.now().year
        current_season_cover_crop = f"{self.current_year - 1}/{self.current_year} COVERCROPS"
        current_season_weeds = f"{self.current_year} WEEDS"
        current_season_cashcrops = f"{self.current_year} CASHCROPS"

        current_seasons = [current_season_cover_crop, current_season_weeds, current_season_cashcrops]
        self.data_current_season = self.data[self.data["Season"].isin(current_seasons)]

        log.info("Season column added successfully.")
        return self.data_current_season


    def tables_state_and_planttype_by_season(self) -> dict:
        """
        Generate a dictionary with years as keys and corresponding data as values.

        Returns:
            dict: Dictionary with years as keys and corresponding DataFrames as values.
        """
        log.info("Generating tables of state and plant type by season.")
        data_by_year = {}
        data_grouped_years = self.data.groupby("Season")

        for year, group in data_grouped_years:
            data_by_year[year] = group.reset_index(drop=True)

        log.debug(f"Data grouped by season: {list(data_by_year.keys())}")
        return data_by_year 
        
    def plot_unique_samples_state_plant_current_season(self) -> None:
        """
        Generate a bar plot showing the distribution of unique MasterRefIDs by state and plant type in the current season.
        """
        log.info("Generating bar plot for unique samples by state and plant type for the current season.")
        
        # Define current season labels
        current_year = self.current_year
        last_year = current_year - 1
        cover_crop_label = f"{last_year}/{current_year} COVERCROPS"
        weeds_label = f"{current_year} WEEDS"
        cash_crops_label = f"{current_year} CASHCROPS"
        
        # Define plant type palette with new labels
        self.planttype_palette = {
            cover_crop_label: "#4C72B0",
            weeds_label: "#55A868",
            cash_crops_label: "#C44E52",
        }
        
        unique_ids_count = (
            self.data_current_season.groupby(["UsState", "Season"])["MasterRefID"]
            .nunique()
            .reset_index()
        )
        
        # Ensure all states are included in the plot
        states_list = ["AL", "DV", "GA", "IL", "KS", "MD", "MS", "NC", "NC01", "TX", "TX01", "TX02", "VA"]
        existing_states = set(unique_ids_count["UsState"])
        missing_states = [state for state in states_list if state not in existing_states]
        
        missing_states_df = pd.DataFrame({"UsState": missing_states})
        self.unique_ids_count_all_states = pd.concat([unique_ids_count, missing_states_df], ignore_index=True).sort_values(by="UsState")
        
        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            bar_plot = sns.barplot(
                data=self.unique_ids_count_all_states,
                x="UsState",
                y="MasterRefID",
                hue="Season",
                palette=self.planttype_palette,
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
            save_path = f"{self.cfg.report.report_plots}/unique_masterrefids_by_state_and_planttype_current_season.png"
            fig.savefig(save_path, dpi=300)
        
        log.info("Unique MasterRefIDs by state and plant type for current season plot saved.")

    def plot_unique_samples_species_current_season(self) -> None:
        """
        Generate a bar plot showing the distribution of unique MasterRefIDs by species in the current season.
        """
        log.info("Generating bar plot for unique samples by species for the current season.")
        
        unique_ids_count = (
            self.data_current_season.groupby(["UsState", "Species"])["MasterRefID"]
            .nunique()
            .reset_index(name="sample_count")
            .sort_values(by="sample_count")
        )

        log.debug(f"Unique IDs Count DataFrame with Color: {unique_ids_count.head()}")

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
            ax.set_title(f"{self.current_year} Samples by Species and Plant Type")

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
            save_path = f"{self.cfg.report.report_plots}/unique_masterrefids_by_species_and_planttype_current_season.png"
            fig.savefig(save_path, dpi=300)
        log.info("Species distribution for current season plot saved.")

    def plot_image_vs_raws_by_species_current_season(self):
        # Count the number of unique Images for each UsState and Extension
        unique_ids_count = (
            self.data_current_season.groupby(["UsState", "Extension"])["Name"]
            .nunique()
            .reset_index()
        )

         # Ensure all states are included in the plot
        states_list = ["AL", "DV", "GA", "IL", "KS", "MD", "MS", "NC", "NC01", "TX", "TX01", "TX02", "VA"]
        existing_states = set(unique_ids_count["UsState"])
        missing_states = [state for state in states_list if state not in existing_states]
        
        missing_states_df = pd.DataFrame({"UsState": missing_states})
        self.unique_ids_count_all_states = pd.concat([unique_ids_count, missing_states_df], ignore_index=True).sort_values(by="UsState")

        # Plotting
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))

            sns.barplot(
                data=self.unique_ids_count_all_states,
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
                f"{self.cfg.report.report_plots}/image_vs_raws_by_species_current_season.png"
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
    plots_season.add_season_column()
    plots_season.plot_unique_samples_state_plant_current_season()
    plots_season.plot_unique_samples_species_current_season()
    plots_season.plot_image_vs_raws_by_species_current_season()
    log.info(f"Task {cfg.general.task} completed.")
