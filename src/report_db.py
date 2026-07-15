#!/usr/bin/env python3
# fmt: off
# isort: off
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from db.connection import get_connection
from db.locations import all_known_locations, batch_folder_regex, roll_up_to_parent
from db.reporting import load_report_dataframe
from plotting import ensure_palette_covers, plot_unique_samples

log = logging.getLogger(__name__)

"""
    report.py's all-years reporting/plotting, sourced from images/samples/locations
    in the DB instead of find_most_recent_data_csv. This is the live path
    (cfg.pipeline): python main.py general.task=report_db +pipeline=[report_db]
"""


class PreprocessingCheckDb:
    """Same NFS directory scan report.py's PreprocessingCheck did, but the
    batch-folder pattern is generated from the DB's real location codes
    (db/locations.py) instead of a hardcoded '2 letters, optionally + 2 digits'
    regex."""

    def __init__(self, cfg: DictConfig, conn) -> None:
        self.conn = conn
        self.storage_path = Path(cfg.paths.longterm_storage)
        if not self.storage_path.exists():
            log.error(f"Path {self.storage_path} does not exist.")
            raise FileNotFoundError(f"Path {self.storage_path} does not exist.")
        log.info(f"Initialized PreprocessingCheckDb for path: {self.storage_path}")

        self.save_table_path = Path(cfg.paths.preprocessing_analysis)
        self.save_plot_dir = Path(cfg.paths.plots_all_years)

    def analyze_directory(self) -> pd.DataFrame:
        results = []
        pattern = batch_folder_regex(all_known_locations(self.conn))

        for subdir in self.storage_path.iterdir():
            if subdir.is_dir() and pattern.match(subdir.name):
                jpg_count, raw_count = self._count_images(subdir)
                folder_metadata = self._get_folder_metadata(subdir)
                results.append({
                    "FolderName": subdir.name,
                    "JPGCount": jpg_count,
                    "RAWCount": raw_count,
                    "CreationDate": folder_metadata[0],
                    "LastModifiedDate": folder_metadata[1],
                })
                log.info(f"Processed folder: {subdir.name} - JPG: {jpg_count}, RAW: {raw_count}")
            else:
                log.info(f"Skipped folder: {subdir.name} (does not match a known location code)")

        df = pd.DataFrame(results)
        log.info(f"Directory analysis completed. Processed {len(results)} folders.")
        return df

    def _count_images(self, folder: Path) -> Tuple[int, int]:
        jpg_count = sum(1 for f in folder.rglob("*.jpg"))
        raw_count = sum(1 for f in folder.rglob("*.ARW"))
        log.debug(f"Counted {jpg_count} JPG files and {raw_count} RAW files in folder: {folder}")
        return jpg_count, raw_count

    def _get_folder_metadata(self, folder: Path) -> Tuple[str, str]:
        stat = folder.stat()
        creation_date = datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d")
        last_modified_date = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        log.debug(f"Retrieved metadata for folder {folder}: CreationDate={creation_date}, LastModifiedDate={last_modified_date}")
        return creation_date, last_modified_date

    def save_to_csv(self, df: pd.DataFrame) -> None:
        df.to_csv(self.save_table_path, index=False)
        log.info(f"Results saved to CSV: {self.save_table_path}")

    def plot_batches_per_week(self, df: pd.DataFrame) -> None:
        """Plots valid (JPGCount == RAWCount) batches created per week, listing
        any unequal-count folders in a text box."""
        df["IsEqual"] = df["JPGCount"] == df["RAWCount"]
        valid_folders = df.loc[df["IsEqual"]].copy()
        invalid_folders = df.loc[~df["IsEqual"]]

        if valid_folders.empty:
            log.info("No valid folders to plot (no folders where JPGCount equals RAWCount).")
            return

        valid_folders.loc[:, "LastModifiedDate"] = pd.to_datetime(valid_folders["LastModifiedDate"])
        valid_folders.set_index("LastModifiedDate", inplace=True)
        valid_folders.loc[:, "WeekStart"] = valid_folders.index.to_period("W").start_time

        batches_per_week = valid_folders.groupby("WeekStart").size().reset_index(name="BatchCount")
        sns.catplot(x="WeekStart", y="BatchCount", data=batches_per_week, kind="bar", height=6, aspect=2)
        plt.title("Number of Preprocessed Batches Created Per Week")
        plt.xlabel("Week Start Date")
        plt.ylabel("Number of Preprocessed Batches")
        plt.xticks(rotation=45)
        plt.tight_layout()

        if not invalid_folders.empty:
            invalid_batch_list = "\n".join(invalid_folders["FolderName"].tolist())
            plt.gcf().text(0.2, 0.95, f"Batches with unequal JPG and RAW counts:\n{invalid_batch_list}",
                            ha="center", va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.5))

        save_path = Path(self.save_plot_dir, "preprocessed_batches_per_week.png")
        plt.savefig(save_path)
        plt.close()
        log.info(f"Plot saved to {save_path}")


class BatchReportDb:
    """Same reports/plots as report.py's BatchReport, sourced from the DB. NC01 is
    rolled up into NC once, up front (db/locations.roll_up_to_parent), instead of
    each plot independently excluding it, renaming it, or leaving it alone."""

    def __init__(self, cfg: DictConfig, conn) -> None:
        self.cfg = cfg
        self.conn = conn
        self.locations = all_known_locations(conn)
        self.known_states = [loc.code for loc in self.locations if loc.parent_code is None]

        self.df = load_report_dataframe(conn)
        self.df["UsState"] = self.df["UsState"].apply(lambda code: roll_up_to_parent(self.locations, code) if pd.notna(code) else code)

        self.config_report_dir()
        self.planttype_palette = {
            "WEEDS": "#55A868",
            "COVERCROPS": "#4C72B0",
            "CASHCROPS": "#C44E52",
            "SOILS": "#A95609",
        }
        self.num_past_days_for_report = cfg.inspection.num_past_days_for_report

    def config_report_dir(self) -> None:
        self.report_dir = Path(self.cfg.paths.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)
        self.reportplot_dir = Path(self.cfg.paths.report_plots)
        self.reportplot_dir.mkdir(exist_ok=True, parents=True)
        self.plots_all_years = Path(self.cfg.paths.plots_all_years)
        self.plots_all_years.mkdir(exist_ok=True, parents=True)

    def write_missing_raws(self, df: pd.DataFrame) -> None:
        columns = [
            "Name", "UsState", "PlantType", "Species", "MasterRefID", "BaseName",
            "Extension", "UploadDateUTC", "ImageIndex", "Username", "HasMatchingJpgAndRaw",
        ]
        df["UploadDateTimeUTC"] = pd.to_datetime(df["UploadDateTimeUTC"])
        df["UploadDateUTC"] = df["UploadDateTimeUTC"].dt.date
        df = df[df["HasMatchingJpgAndRaw"] == False][columns].reset_index(drop=True)
        df.to_csv(self.cfg.paths.missing_batch_folders, index=False)
        log.info("Missing raws data written successfully.")

    def num_uploads_selected_days_by_state(self) -> None:
        df = self.df.copy()
        df["UploadDateTimeUTC"] = pd.to_datetime(df["UploadDateTimeUTC"])
        df["UploadDateUTC"] = pd.to_datetime(df["UploadDateTimeUTC"].dt.date)
        current_date_time = pd.to_datetime(datetime.now().date())
        selected_days_ago = pd.to_datetime(current_date_time - timedelta(self.num_past_days_for_report))

        df_last_selected_days = df[
            (df["UploadDateUTC"] >= selected_days_ago) & (df["UploadDateUTC"] <= current_date_time)
        ].copy()
        df_last_selected_days["IsDuplicated"] = df_last_selected_days.duplicated("Name", keep=False)
        grouped = (
            df_last_selected_days.groupby(
                ["UsState", "PlantType", "Species", "Extension", "HasMatchingJpgAndRaw", "IsDuplicated"]
            ).size().reset_index(name="count")
        )
        file_name = f"uploads_last_{self.num_past_days_for_report}_days_by_state.csv"
        grouped.to_csv(Path(self.cfg.paths.reportdir_timestamp, file_name), index=False)
        log.info(f"Created table of uploads from last {selected_days_ago} days by location successfully.")

    def plot_unique_masterrefids_by_state_and_planttype(self) -> None:
        plot_unique_samples(
            self.df,
            hue_col="PlantType",
            palette=self.planttype_palette,
            title="Unique MasterRefIDs (samples) by State and Plant Type",
            save_path=f"{self.plots_all_years}/unique_masterrefids_by_state_and_planttype.png",
            known_states=self.known_states,
        )
        log.info("Unique MasterRefIDs plot saved.")

    def plot_image_vs_raws_by_species(self) -> None:
        data = self.df
        unique_ids_count = data.groupby(["UsState", "Extension"])["Name"].nunique().reset_index()

        existing = set(unique_ids_count["UsState"])
        missing_rows = [
            {"UsState": state, "Extension": ext, "Name": 0}
            for state in self.known_states if state not in existing
            for ext in ("jpg", "arw")
        ]
        unique_ids_count = pd.concat(
            [unique_ids_count, pd.DataFrame(missing_rows)], ignore_index=True
        ).sort_values(by="UsState")

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=unique_ids_count, x="UsState", y="Name", hue="Extension", ax=ax)
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title("Number of Images by State and by Image Extension")
            ax.set_ylabel("Number of Images")
            ax.set_xlabel("State Location")
            ax.legend(title="Image Type")
            fig.tight_layout()
            save_path = f"{self.plots_all_years}/image_vs_raws_by_species.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            log.info("Jpg vs Raws plot saved.")

    def plot_sample_species_distribution(self) -> None:
        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]
        data = data[data["Extension"] == "arw"]
        samplecount_df = (
            data.groupby(["PlantType", "Species"])["MasterRefID"]
            .nunique().reset_index(name="sample_count").sort_values(by="sample_count")
        )
        palette = ensure_palette_covers(self.planttype_palette, samplecount_df["PlantType"])
        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(8, 14))
            sns.barplot(samplecount_df, y="Species", x="sample_count", hue="PlantType", palette=palette, ax=ax)
            ax.set_ylabel("Species")
            ax.set_xlabel("Number of Unique Samples")
            ax.text(-0.050, -0.035, "$^{*}$HasMatchingJpgAndRaw = True", ha="center", fontsize=9, transform=ax.transAxes)
            ax.figure.suptitle("Samples by Species and Plant Type", fontsize=18)
            for p in ax.patches:
                width = p.get_width()
                ax.text(width + 1, p.get_y() + p.get_height() / 2, "{:1.0f}".format(width), ha="left", va="center")
            fig.tight_layout()
            save_path = f"{self.plots_all_years}/unique_masterrefids_by_species_and_planttype.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            log.info("Species Distribution plot saved.")

    def plot_cumulative_samples_species_by_year(self) -> None:
        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]
        data = data[data["Extension"] == "arw"]

        data["CameraInfo_DateTime"] = pd.to_datetime(data["CameraInfo_DateTime"])
        data["Year"] = data["CameraInfo_DateTime"].dt.year
        data["Year"] = data["Year"].replace(2021, 2022)

        samplecount_df = (
            data.groupby(["UsState", "Species", "Year"])["MasterRefID"]
            .nunique().reset_index(name="sample_count").sort_values(by="Year")
        )
        samplecount_df["cumulative_count"] = samplecount_df.groupby(["UsState", "Species"])["sample_count"].cumsum()
        samplecount_pivot = samplecount_df.pivot_table(
            index=["Species", "UsState"], columns="Year", values="sample_count", aggfunc="sum", fill_value=0
        )
        samplecount_pivot = samplecount_pivot.sort_index(level="Species")

        for state in samplecount_df["UsState"].unique():
            fig, ax = plt.subplots(figsize=(10, 7))
            state_data = samplecount_pivot.xs(state, level="UsState")
            state_data = state_data.sort_index(level="Species")
            state_data.plot(kind="barh", stacked=True, ax=ax, cmap="tab20")
            ax.set_xlabel("Cumulative Number of Unique Samples")
            ax.set_ylabel("Species")
            ax.set_title(f"Cumulative Samples by Species and Year: {state}", fontsize=18)
            ax.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
            fig.tight_layout()
            save_path = f"{self.plots_all_years}/cumulative_stacked_samples_by_species_for_{state}.png"
            fig.savefig(save_path, dpi=300)
            log.info(f"Cumulative Samples by Species plot saved for {state}.")
            plt.close(fig)

    def plot_sample_species_state_distribution(self) -> None:
        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]
        data = data[data["Extension"] == "arw"]

        samplecount_df = (
            data.groupby(["UsState", "PlantType", "Species"])["MasterRefID"]
            .nunique().reset_index(name="sample_count").sort_values(by="sample_count")
        )
        palette = ensure_palette_covers(self.planttype_palette, samplecount_df["PlantType"])

        for state in samplecount_df["UsState"].unique():
            fig, ax = plt.subplots(figsize=(8, 6))
            state_data = samplecount_df[samplecount_df["UsState"] == state]
            sns.barplot(data=state_data, x="sample_count", y="Species", hue="PlantType", palette=palette, ax=ax)
            for p in ax.patches:
                width = p.get_width()
                ax.text(width + 1, p.get_y() + p.get_height() / 2, "{:1.0f}".format(width), ha="left", va="center")
            ax.set_ylabel("Species")
            ax.set_xlabel("Number of Unique Samples")
            ax.text(-0.050, -0.085, "$^{*}$HasMatchingJpgAndRaw = True", ha="center", fontsize=9, transform=ax.transAxes)
            ax.figure.suptitle(f"Samples by Species: {state}", fontsize=18)
            fig.tight_layout()
            save_path = f"{self.plots_all_years}/unique_masterrefids_by_species_for_{state}.png"
            fig.savefig(save_path, dpi=300)
            log.info("Species Distribution plot saved.")
            plt.close(fig)

    def plot_num_samples_season(self) -> None:
        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]
        unique_ids_count = data.groupby(["PlantType"])["MasterRefID"].nunique().reset_index().sort_values(by="MasterRefID")

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))
            bar_plot = sns.barplot(data=unique_ids_count, x="PlantType", y="MasterRefID", ax=ax, width=0.25)
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.figure.suptitle("Samples by Species", fontsize=18)
            ax.annotate("$^{*}$HasMatchingJpgAndRaw = True", xy=(0.05, 0.9), xycoords="axes fraction", ha="left", va="bottom", annotation_clip=False)
            ax.set_ylabel("# MasterRefIDs (samples)")
            ax.set_xlabel("Plant Type")
            for bar_container in bar_plot.containers:
                ax.bar_label(bar_container, label_type="edge", padding=3, fontsize=7)
            fig.tight_layout()
            save_path = f"{self.plots_all_years}/unique_masterrefids_by_season.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            log.info("Unique MasterRefIDs by Plant Type plot saved.")

    def plot_num_samples_usstate(self) -> None:
        data = self.df[self.df["HasMatchingJpgAndRaw"] == True]
        unique_ids_count = data.groupby(["UsState"])["MasterRefID"].nunique().reset_index()

        existing = set(unique_ids_count["UsState"])
        missing = [{"UsState": state, "MasterRefID": 0} for state in self.known_states if state not in existing]
        unique_ids_count = pd.concat([unique_ids_count, pd.DataFrame(missing)], ignore_index=True).sort_values(by="MasterRefID")

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 6))
            bar_plot = sns.barplot(data=unique_ids_count, x="UsState", y="MasterRefID", ax=ax, width=0.25)
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.figure.suptitle("Samples by State", fontsize=18)
            ax.annotate("$^{*}$HasMatchingJpgAndRaw = True", xy=(0.05, 0.9), xycoords="axes fraction", ha="left", va="bottom", annotation_clip=False)
            ax.set_ylabel("# MasterRefIDs (samples)")
            ax.set_xlabel("State")
            for bar_container in bar_plot.containers:
                ax.bar_label(bar_container, label_type="edge", padding=3, fontsize=7)
            fig.tight_layout()
            save_path = f"{self.plots_all_years}/unique_masterrefids_by_state.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            log.info("Unique MasterRefIDs by UsState plot saved.")


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    conn = get_connection(cfg.paths.db_path)
    try:
        batchrep = BatchReportDb(cfg, conn)
        batchrep.write_missing_raws(batchrep.df)
        batchrep.plot_unique_masterrefids_by_state_and_planttype()
        batchrep.plot_sample_species_distribution()
        batchrep.plot_image_vs_raws_by_species()
        batchrep.plot_num_samples_season()
        batchrep.plot_num_samples_usstate()
        batchrep.plot_sample_species_state_distribution()
        batchrep.plot_cumulative_samples_species_by_year()
        batchrep.num_uploads_selected_days_by_state()

        analyzer = PreprocessingCheckDb(cfg, conn)
        analysis_results_df = analyzer.analyze_directory()
        analyzer.plot_batches_per_week(analysis_results_df)
        analyzer.save_to_csv(analysis_results_df)
    finally:
        conn.close()
    log.info(f"{cfg.general.task} completed.")
