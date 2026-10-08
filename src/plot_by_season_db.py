#!/usr/bin/env python3
# fmt: off
# isort: off
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig

from db.connection import get_connection
from db.locations import all_known_locations, roll_up_to_parent
from db.reporting import load_report_dataframe
from plotting import plot_unique_samples

log = logging.getLogger(__name__)

"""
    plot_by_season.py's current-season reporting/plotting, sourced from
    images/samples/locations in the DB instead of the permanent CSV. Not part of
    the automatic pipeline (cfg.pipeline) yet - run manually and compare its
    aggregate counts against plot_by_season.py's:
        python main.py general.task=plot_by_season_db +pipeline=[plot_by_season_db]
"""


class PlotsBySeasonDb:
    """Same current-season plots as plot_by_season.py's PlotsBySeason, sourced
    from the DB. NC01 is rolled up into NC once, up front, instead of being
    excluded entirely for this one plot's data (the third of the three
    inconsistent NC01 treatments the plan calls out)."""

    def __init__(self, cfg: DictConfig, conn) -> None:
        log.info("Initializing PlotsBySeasonDb class.")
        self.conn = conn
        self.locations = all_known_locations(conn)
        self.known_states = [loc.code for loc in self.locations if loc.parent_code is None]
        self.current_year = datetime.now().year

        self.report_dir = Path(cfg.paths.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)
        self.reportplot_dir = Path(cfg.paths.report_plots)
        self.reportplot_dir.mkdir(exist_ok=True, parents=True)
        self.plots_current_season = Path(cfg.paths.plots_current_season)
        self.plots_current_season.mkdir(exist_ok=True, parents=True)

        self.df = load_report_dataframe(conn)
        self.df["UsState"] = self.df["UsState"].apply(lambda code: roll_up_to_parent(self.locations, code) if pd.notna(code) else code)
        self.df["CameraInfo_DateTime"] = pd.to_datetime(self.df["CameraInfo_DateTime"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
        self.df = self.df.dropna(subset=["CameraInfo_DateTime"])

    def fill_missing_camera_datetime(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fill missing CameraInfo_DateTime values using a JPG with the same Stem.

        Existing CameraInfo_DateTime values are preserved.
        """
        jpg_mask = (
            df["CameraInfo_DateTime"].notna()
            & df["Extension"].str.casefold().eq("jpg")
        )

        datetime_lookup = (
            df.loc[jpg_mask, ["Stem", "CameraInfo_DateTime"]]
            .drop_duplicates(subset="Stem", keep="first")
            .set_index("Stem")["CameraInfo_DateTime"]
        )

        df = df.copy()

        missing_mask = df["CameraInfo_DateTime"].isna()

        df.loc[missing_mask, "CameraInfo_DateTime"] = (
            df.loc[missing_mask, "Stem"].map(datetime_lookup)
        )

        return df

    def add_season_column(self) -> pd.DataFrame:
        log.info("Adding 'Season' column to the data.")

        self.df = self.fill_missing_camera_datetime(self.df)

        self.df["CameraInfo_DateTime"] = pd.to_datetime(
            self.df["CameraInfo_DateTime"],
            errors="coerce",
        )

        date_time = self.df["CameraInfo_DateTime"]
        plant_type = self.df["PlantType"].astype("string").str.upper()
        year = date_time.dt.year.astype("Int64")

        # Default for weeds, cash crops, and soil collections.
        self.df["Season"] = (
            year.astype("string")
            + " "
            + plant_type
        )

        cover_crop_mask = plant_type.eq("COVERCROPS")
        fall_mask = cover_crop_mask & date_time.dt.month.ge(10)
        spring_mask = cover_crop_mask & date_time.dt.month.lt(10)

        self.df.loc[fall_mask, "Season"] = (
            year[fall_mask].astype("string")
            + "/"
            + (year[fall_mask] + 1).astype("string")
            + " COVERCROPS"
        )

        self.df.loc[spring_mask, "Season"] = (
            (year[spring_mask] - 1).astype("string")
            + "/"
            + year[spring_mask].astype("string")
            + " COVERCROPS"
        )

        invalid_mask = date_time.isna() | plant_type.isna()
        self.df.loc[invalid_mask, "Season"] = pd.NA

        current_seasons = [
            f"{self.current_year - 1}/{self.current_year} COVERCROPS",
            f"{self.current_year} WEEDS",
            f"{self.current_year} CASHCROPS",
            f"{self.current_year} SOILS",
        ]

        data_current_season = self.df.loc[
            self.df["Season"].isin(current_seasons)
        ].copy()

        log.info(
            "Season column added successfully. Retained %d of %d rows.",
            len(data_current_season),
            len(self.df),
        )

        return data_current_season

    def plot_unique_samples_state_plant_current_season(self, data_current_season) -> None:
        log.info("Generating bar plot for unique samples by state and plant type for the current season.")
        last_year = self.current_year - 1
        cover_crop_label = f"{last_year}/{self.current_year} COVERCROPS"
        weeds_label = f"{self.current_year} WEEDS"
        cash_crops_label = f"{self.current_year} CASHCROPS"
        soils_label = f"{self.current_year} SOILS"
        planttype_palette = {cover_crop_label: "#4C72B0", weeds_label: "#55A868", cash_crops_label: "#C44E52", soils_label: "#A95609"}

        plot_unique_samples(
            data_current_season,
            hue_col="Season",
            palette=planttype_palette,
            title=f"{self.current_year} Unique MasterRefIDs (samples) by State and Plant Type",
            save_path=f"{self.plots_current_season}/unique_masterrefids_by_state_and_planttype_current_season.png",
            known_states=self.known_states,
            hue_order=[cover_crop_label, weeds_label, cash_crops_label, soils_label],
        )
        log.info("Unique MasterRefIDs by state and plant type for current season plot saved.")

    def plot_unique_samples_species_current_season(self, data_current_season) -> None:
        log.info("Generating bar plot for unique samples by species for the current season.")
        data = data_current_season[data_current_season["HasMatchingJpgAndRaw"] == True].copy()
        unique_ids_count = (
            data.groupby(["Species"])["MasterRefID"].nunique().reset_index(name="sample_count").sort_values(by="sample_count")
        )

        with plt.style.context("ggplot"):
            fig, ax = plt.subplots(figsize=(12, 8))
            x_positions = range(len(unique_ids_count))
            bars = ax.bar(x=x_positions, height=unique_ids_count["sample_count"], color="#C44E52", edgecolor="black", width=0.5)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(unique_ids_count["Species"], rotation=90)
            ax.set_ylabel("Number of Unique Samples")
            ax.set_xlabel("Species")
            ax.text(0.5, 0.8, "$^{*}$HasMatchingJpgAndRaw = True", ha="center", fontsize=9, transform=ax.transAxes)
            ax.set_title(f"{self.current_year} Samples by Species")
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.0f}", ha="center", va="bottom")
            fig.tight_layout()
            save_path = f"{self.plots_current_season}/unique_masterrefids_by_species_current_season.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
        log.info("Species distribution for current season plot saved.")

    def plot_image_vs_raws_by_species_current_season(self, data_current_season) -> None:
        unique_ids_count = data_current_season.groupby(["UsState", "Extension"])["Name"].nunique().reset_index()

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
            sns.barplot(data=unique_ids_count, x="UsState", y="Name", hue="Extension", hue_order=["arw", "jpg"], ax=ax)
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.set_title(f"{self.current_year}: Missing Raw Uploads by State")
            ax.set_ylabel("Number of Images")
            ax.set_xlabel("State Location")
            ax.legend(title="Image Type")
            fig.tight_layout()
            save_path = f"{self.plots_current_season}/image_jpgs_vs_raws_by_state_current_season.png"
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            log.info("Jpg vs Raws plot saved for current season.")


def main(cfg: DictConfig) -> None:
    log.info(f"Starting task: {cfg.general.task}")
    conn = get_connection(cfg.paths.db_path)
    try:
        plots_season = PlotsBySeasonDb(cfg, conn)
        data_current_season = plots_season.add_season_column()
        plots_season.plot_unique_samples_state_plant_current_season(data_current_season)
        plots_season.plot_unique_samples_species_current_season(data_current_season)
        plots_season.plot_image_vs_raws_by_species_current_season(data_current_season)
    finally:
        conn.close()
    log.info(f"Task {cfg.general.task} completed.")
