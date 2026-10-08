import os
import logging
import shutil

from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import DictConfig
from utils.utils import download_from_url, get_exif_data

from db.connection import get_connection
from db.reporting import load_report_dataframe

log = logging.getLogger(__name__)

"""
    image_inspection.py's reviewer-sample-image inspection, sourced from
    images/samples in the DB instead of find_most_recent_data_csv - same
    report_db.py-style migration off the CSV chain.
"""


class InspectRecentUploadsDb:
    """Randomly selects up to num_images_to_inspect images per state uploaded
    in the last num_past_days_to_inspect days, plots each with its EXIF/sample
    metadata, and saves the plots to cfg.paths.inspectdir for reviewer
    inspection."""

    def __init__(self, cfg: DictConfig, conn) -> None:
        self.cfg = cfg
        self.df = load_report_dataframe(conn)
        self.config_inspection_dir()

        self.temp_image_dir = Path(cfg.paths.temp_image_dir)
        self.temp_image_dir.mkdir(exist_ok=True, parents=True)

        self.num_past_days_to_inspect = cfg.inspection.num_past_days_to_inspect
        self.num_images_to_inspect = cfg.inspection.num_images_to_inspect

    def config_inspection_dir(self) -> None:
        log.info("Creating output path for the results")
        self.report_dir = Path(self.cfg.paths.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)

        self.inspect_dir = Path(self.cfg.paths.inspectdir)
        self.inspect_dir.mkdir(exist_ok=True, parents=True)

    def download_images_temp(self) -> None:
        """Download up to selected random images from each state that uploaded for last __ selected days."""
        self.df["upload_date"] = pd.to_datetime(self.df["UploadDateTimeUTC"].str[:10], format="%Y-%m-%d")

        current_date_time = pd.to_datetime(datetime.now().date())
        targeted_days_ago = current_date_time - timedelta(self.num_past_days_to_inspect)

        df_targeted_days = self.df[
            (self.df["upload_date"] >= targeted_days_ago) & (self.df["upload_date"] <= current_date_time)
        ]

        if df_targeted_days.empty:
            log.info(f"No uploads in the last {self.num_past_days_to_inspect} days.")
            return

        df_targeted_days_states = df_targeted_days["UsState"].unique()

        log.info(f"Temporary downloading random photos from each state that uploaded in {self.cfg.paths.temp_image_dir}")

        for state in df_targeted_days_states:
            state_df = df_targeted_days[df_targeted_days["UsState"] == state]
            jpg_df = state_df[state_df["ImageURL"].str.endswith(".JPG")]

            if jpg_df.empty:
                log.info(f"No .JPG images found for state: {state}")
                continue

            num_images = min(len(jpg_df), self.num_images_to_inspect)
            log.info(f"Selecting {num_images} images for state: {state}")

            random_imageurls = jpg_df["ImageURL"].sample(n=num_images, random_state=42, ).tolist()

            for url in random_imageurls:
                try:
                    download_from_url(url, self.cfg.paths.temp_image_dir)
                except Exception as e:
                    log.error(f"Failed to download image from {url}: {e}")

    def plotting_sample_images_and_exif(self) -> None:
        """Plots sample images along with important EXIF information."""
        log.info("Plotting images with exif data of images selected for sampling")
        try:
            [shutil.rmtree(os.path.join(self.inspect_dir, folder_name)) for folder_name in os.listdir(self.inspect_dir)]
        except OSError as e:
            log.error(f"Error while removing existing directories: {e}")

        temp_image_dir = self.cfg.paths.temp_image_dir

        for filename in os.listdir(temp_image_dir):
            if filename.endswith((".JPG", ".jpg", ".jpeg", ".png", ".gif")):
                image_path = os.path.join(temp_image_dir, filename)
                image_name = os.path.basename(image_path)
            else:
                log.info(f"Error: check the files in {temp_image_dir}.")
                continue

            try:
                exif_info = get_exif_data(image_path)
                selected_tags = ["Image DateTime", "EXIF ExposureTime", "EXIF ISOSpeedRatings", "EXIF FNumber", "EXIF FocalLength"]
                selected_info = {tag: value for tag, value in exif_info.items() if tag in selected_tags}

                selected_info["UsState"] = self.df.loc[self.df["Name"] == image_name, "UsState"].iloc[0]
                selected_info["Username"] = self.df.loc[self.df["Name"] == image_name, "Username"].iloc[0]
                selected_info["Species"] = self.df.loc[self.df["Name"] == image_name, "Species"].iloc[0]
                selected_info["UploadDateTimeUTC"] = self.df.loc[self.df["Name"] == image_name, "UploadDateTimeUTC"].iloc[0]
                selected_info["HasMatchingJpgAndRaw"] = self.df.loc[self.df["Name"] == image_name, "HasMatchingJpgAndRaw"].iloc[0]

                image = Image.open(image_path)

                fig, (ax_image, ax_info) = plt.subplots(1, 2, figsize=(8, 3))
                ax_image.imshow(image)
                ax_image.axis("off")
                ax_image.set_title("Sample image with EXIF Data")

                exif_text = "\n".join([f"{tag} : {value}" for tag, value in selected_info.items()])
                ax_info.text(0, 1, exif_text, fontsize=10, color="black", verticalalignment="top")
                ax_info.axis("off")

                state_folder = os.path.join(self.inspect_dir, selected_info["UsState"])
                os.makedirs(state_folder, exist_ok=True)

                plt.savefig(Path(state_folder) / os.path.basename(image_path), dpi=200)
                plt.clf()
                plt.close(fig)

                os.remove(image_path)
            except Exception as e:
                log.error(f"Error plotting images for inspection {image_path}: {e}")


def main(cfg: DictConfig) -> None:
    log.info(f"Starting {cfg.general.task}")
    conn = get_connection(cfg.paths.db_path)
    try:
        imginspect = InspectRecentUploadsDb(cfg, conn)
        imginspect.download_images_temp()
        imginspect.plotting_sample_images_and_exif()
    finally:
        conn.close()
    log.info(f"{cfg.general.task} completed.")
