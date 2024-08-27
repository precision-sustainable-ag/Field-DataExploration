import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from utils.utils import find_most_recent_data_csv

log = logging.getLogger(__name__)


class BatchReport:
    """A class to generate reports for Field data visualization,
    focusing on different aspects of plant data.
    """

    def __init__(self, cfg) -> None:
        """Initialize the BatchReport with configuration and data loading."""
        self.cfg = cfg
        self.csv_path = find_most_recent_data_csv(cfg.data.datadir)
        self.df = self.read()
        self.config_report_dir()
        self.config_palettes()
        # self.cfg.reportdir = cfg.report.reportdir

    def config_palettes(self) -> None:
        """Configure the color palettes for different plant types."""
        self.planttype_palette = {
            "WEEDS": "#55A868",
            "COVERCROPS": "#4C72B0",
            "CASHCROPS": "#C44E52",
        }

    def config_report_dir(self) -> None:
        """Configure and create necessary directories for report outputs."""
        self.report_dir = Path(self.cfg.report.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)

        self.reportplot_dir = Path(self.cfg.report.report_plots)
        self.reportplot_dir.mkdir(exist_ok=True, parents=True)

        self.plots_all_years = Path(self.cfg.report.plots_all_years)
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
        df.to_csv(self.cfg.report.missing_batch_folders, index=False)
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

        grouped_df_last_7_days.to_csv(self.cfg.report.uploads_7days, index=False)
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
            save_path = f"{self.cfg.report.plots_all_years}/unique_masterrefids_by_state_and_planttype.png"
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
                f"{self.cfg.report.plots_all_years}/image_vs_raws_by_species.png"
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
            save_path = f"{self.cfg.report.plots_all_years}/unique_masterrefids_by_species_and_planttype.png"
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
            save_path = f"{self.cfg.report.plots_all_years}/cumulative_stacked_samples_by_species_for_{state}.png"
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
            # plt.subplots_adjust(top=0.93)
            save_path = f"{self.cfg.report.plots_all_years}/unique_masterrefids_by_species_for_{state}.png"
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
                f"{self.cfg.report.plots_all_years}/unique_masterrefids_by_season.png"
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
                f"{self.cfg.report.plots_all_years}/unique_masterrefids_by_state.png"
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
    log.info(f"{cfg.general.task} completed.")
