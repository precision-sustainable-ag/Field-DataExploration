#!/usr/bin/env python3
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from utils.utils import find_most_recent_csv

log = logging.getLogger(__name__)


class PlotRawImages:
    """
    This class handles the preprocessing of image metadata and plotting of the species distribution.

    Attributes:
        main_directory_path (Path): Path to the main directory containing the data files.
        csv_filename (str): Filename of the CSV file to process.
        plot_savedir (Path): Directory where the plots will be saved.
        plot_savepath (Path): Full path for saving the plot.
        dpi (int): Resolution of the saved plot in dots per inch.
        transparency (bool): Flag to set the background of the plot as transparent.
        df (pd.DataFrame): DataFrame loaded with the most recent data from the CSV file.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initializes the PlotRawImages object with configuration settings."""
        self.main_directory_path = Path(cfg.data.processed_datadir).parent
        self.csv_filename = "merged_blobs_tables_metadata.csv"
        self.plot_savedir = Path(cfg.report.report_plots)
        self.plot_savedir.mkdir(exist_ok=True, parents=True)
        self.plot_savepath = Path(self.plot_savedir, "raw_images_by_species.png")
        self.dpi = 300
        self.transparency = False

        self.df = self._get_most_recent_table()

    def _get_most_recent_table(self):
        """Finds and loads the most recent CSV data file based on naming conventions and directory structure.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        most_recent_csv_path = find_most_recent_csv(
            self.main_directory_path, self.csv_filename
        )
        return pd.read_csv(most_recent_csv_path)

    def preprocess_df(self, df):
        """
        Preprocesses the data by filtering and organizing it to facilitate plotting.

        Parameters:
            df (pd.DataFrame): The DataFrame to preprocess. Note: this argument is currently unused.

        Returns:
            pd.DataFrame: The processed DataFrame ready for plotting.
        """
        df = self.df.copy()
        # Extract base name and extension

        df["base_name"] = df["name"].str.rsplit(".", n=1).str[0]
        df["extension"] = df["name"].str.split(".", n=1).str[-1]
        df["extension"] = df["extension"].str.lower()

        # Filter out only the rows with 'jpg' or 'arw' extensions to reduce processing
        filtered_df = df[df["extension"].isin(["jpg", "arw"])]

        # Identify matching files
        # Create a grouped DataFrame by 'base_name' and check if both 'jpg' and 'arw' are present for each group
        matches = filtered_df.groupby("base_name")["extension"].apply(
            lambda x: "jpg" in x.values and "arw" in x.values
        )
        # # Convert 'matches' Series back to a DataFrame and merge with the original DataFrame on 'base_name'
        matches_df = matches.reset_index().rename(
            columns={"extension": "has_matching_jpg_and_raw"}
        )
        dfmatch = pd.merge(df, matches_df, on="base_name", how="left")
        # # Fill NaN values in 'has_matching_file' with False for files that are neither 'jpg' nor 'arw'
        dfmatch["has_matching_jpg_and_raw"] = dfmatch[
            "has_matching_jpg_and_raw"
        ].fillna(False)
        df_only_matching = dfmatch[dfmatch["has_matching_jpg_and_raw"] == True]

        statedf = (
            df_only_matching.groupby(["PlantType", "Species", "extension"])["Species"]
            .value_counts()
            .reset_index()
        )
        statedf = statedf[statedf["extension"] == "arw"]
        statedf = statedf.sort_values(
            by=["PlantType", "count"], ascending=[False, True]
        )
        return statedf

    def plot_species_distribution(self, df, save=False):
        """
        Plots the distribution of species within plant types.

        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to plot.
            save (bool, optional): Whether to save the plot to a file. Defaults to False.
        """
        with plt.style.context("ggplot"):
            g = sns.catplot(
                data=df,
                y="Species",
                x="count",
                # hue="extension",
                kind="bar",
                # hue="Combined",  # Assign 'PlantType' to hue for color coding
                col="PlantType",
                height=8,
                sharey=False,
                sharex=True,
                # legend=False,
                aspect=1,
            )
            # To rename the subplot titles for 'state_id'
            state_id_mapping = {
                # Assuming you have a mapping of state_id to more descriptive names
                "WEEDS": "Weeds",
                "CASHCROPS": "Cash Crops",
                "COVERCROPS": "Cover Crops",
            }
            # Iterate over the axes titles and set new titles
            for ax in g.axes.flatten():
                bars = ax.patches
                # Extract the state_id from the title
                state_id = ax.title.get_text().split(" = ")[-1].replace("'", "")
                if state_id in state_id_mapping:
                    # Set the new title with a more descriptive name
                    ax.set_title(state_id_mapping[state_id])
                else:
                    # Optionally handle cases where state_id is not in your mapping
                    ax.set_title("Unknown State")
                for i, bar in enumerate(bars):
                    # Calculate a slight variation in color for each bar
                    if len(ax.patches) < 4:
                        palette = sns.color_palette("Reds", n_colors=len(bars))
                        bar.set_color(palette[i])
                    elif 4 < len(ax.patches) < 10:
                        palette = sns.color_palette("Blues", n_colors=len(bars))
                        bar.set_color(palette[i])
                    elif 25 < len(ax.patches) < 35:
                        palette = sns.color_palette("BuGn", n_colors=len(bars))
                        bar.set_color(palette[i])
                g.set_axis_labels("", "")

            g.figure.suptitle("Raw Images by Common Name", fontsize=18)
            # Adjust the layout to make room for the title
            plt.subplots_adjust(top=0.87)
            if save:
                plt.savefig(
                    self.plot_savepath,
                    dpi=self.dpi,
                    transparent=self.transparency,
                    bbox_inches="tight",
                )


def main(cfg: DictConfig) -> None:
    """
    Main function to execute the plotting task based on the provided configuration.

    Parameters:
        cfg (DictConfig): Configuration settings.
    """
    log.info(f"Starting {cfg.general.task}")
    # Example usage:
    processor = PlotRawImages(cfg)
    df = processor.preprocess_df(processor.df)
    processor.plot_species_distribution(df, save=True)
