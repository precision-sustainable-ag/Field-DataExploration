import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from utils.utils import find_most_recent_data_csv

log = logging.getLogger(__name__)


class BatchReport:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.csv_path = find_most_recent_data_csv(cfg.data.datadir)
        self.df = self.read()
        self.config_report_dir()
        self.config_palettes()
        # self.cfg.reportdir = cfg.report.reportdir

    def config_palettes(self):
        self.planttype_palette = {
            "WEEDS": "#55A868",
            "COVERCROPS": "#4C72B0",
            "CASHCROPS": "#C44E52",
        }

    def config_report_dir(self):
        self.report_dir = Path(self.cfg.report.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)

        self.reportplot_dir = Path(self.cfg.report.report_plots)
        self.reportplot_dir.mkdir(exist_ok=True, parents=True)

    def read(self) -> pd.DataFrame:
        log.info("Reading and converting datetime columns in CSV")
        log.info(f"Reading path: {self.csv_path}")
        df = pd.read_csv(self.csv_path, dtype={"SubBatchIndex": str})
        return df

    def missing_raws(self, df):
        # By state, species, upload_time
        columns = [
            "Name",
            "UsState",
            "PlantType",
            "Species",
            "MasterRefID",
            "BaseName",
            "ImageIndex",
            "Username",
            "HasMatchingJpgAndRaw",
        ]
        df = (
            df[df["HasMatchingJpgAndRaw"] == False][columns]
            .drop_duplicates(subset="Name")
            .reset_index(drop=True)
        )
        df.to_csv(self.cfg.report.missing_batch_folders)

    def plot_unique_masterrefids_by_state(self):
        data = self.df.copy()
        # Filter the data for JPG and RAW images
        filtered_data = data[data["Extension"].isin(["jpg", "arw"])]

        # Count the number of unique MasterRefID for each UsState and Extension
        unique_ids_count = (
            filtered_data.groupby(["UsState", "Extension"])["MasterRefID"]
            .nunique()
            .reset_index()
        )
        unique_ids_count.rename(
            columns={"MasterRefID": "UniqueMasterRefIDs"}, inplace=True
        )

        # Plotting

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))

            sns.barplot(
                data=unique_ids_count,
                x="UsState",
                y="UniqueMasterRefIDs",
                hue="Extension",
                ax=ax,
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title(
                "Number of Unique MasterRefIDs by JPGs and RAW Images by State Location"
            )
            ax.set_ylabel("Number of Unique MasterRefIDs")
            ax.set_xlabel("State Location")
            ax.legend(title="Image Type")
            fig.tight_layout()
            save_path = (
                f"{self.cfg.report.report_plots}/unique_masterrefids_by_state.png"
            )
            fig.savefig(save_path, dpi=300)

    def plot_sample_species_distribution(self):
        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]
        data = data[data["Extension"] == "arw"]

        samplecount_df = (
            data.groupby(["PlantType", "Species"])["MasterRefID"]
            .nunique()
            .reset_index(name="sample_count")
            .sort_values(by="sample_count")
        )
        print(samplecount_df)
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
            save_path = (
                f"{self.cfg.report.report_plots}/unique_masterrefids_by_species.png"
            )
            fig.savefig(save_path, dpi=300)


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    batchrep = BatchReport(cfg)
    batchrep.missing_raws(batchrep.df)
    batchrep.plot_unique_masterrefids_by_state()
    batchrep.plot_sample_species_distribution()
    log.info(f"{cfg.general.task} completed.")
