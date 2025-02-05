import logging
import os
import cv2

from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from utils.utils import find_most_recent_data_csv, download_from_url, get_exif_data

log = logging.getLogger(__name__)

class InsepctRecentUploads:
    """This class randomly select upto 15 images(less than 15 if the total uploaded images are less),
    plot images and relevant metadata, save plots to reports so they can be inspected."""

    def __init__(self, cfg) -> None:
        """Initialize the InsepctRecentUploads class with confguration and data loading"""
        self.cfg = cfg
        self.csv_path = find_most_recent_data_csv(cfg.data.datadir)
        self.df = self.read()
        self.config_inspection_dir()

        self.temp_image_dir = Path(self.cfg.temp.temp_image_dir)
        self.temp_image_dir.mkdir(exist_ok=True, parents=True)

        self.days = 180
        self.num_images_to_inspect = 5

    def read(self) -> pd.DataFrame:
        """Read and load data from a CSV file."""
        log.info(f"Reading path: {self.csv_path}")
        df = pd.read_csv(self.csv_path, dtype={"SubBatchIndex": str}, low_memory=False)
        return df

    def config_inspection_dir(self) -> None:
        """Create directory for inspecting samples."""
        log.info(f"Creating output path for the results")
        self.report_dir = Path(self.cfg.report.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)

        self.inspect_dir = Path(self.cfg.report.inspectdir)
        self.inspect_dir.mkdir(exist_ok=True, parents=True)

    def download_images_temp(self) -> None:
        """Download up to 5 random images from each state that uploaded in last 7 days."""

        df = self.df
        
        # Extract the date from UploadDateTimeUTC
        df['upload_date'] = pd.to_datetime(df['UploadDateTimeUTC'].str[:10], format='%Y-%m-%d')

        # Define the date range for the last 7 days
        current_date_time = pd.to_datetime(datetime.now().date())
        seven_days_ago = current_date_time - timedelta(self.days)

        # Filter the DataFrame for the last 7 days
        df_last_7_days = df[(df['upload_date'] >= seven_days_ago) & (df['upload_date'] <= current_date_time)]
        
        if df_last_7_days.empty:
            log.info("No uploads in the last 7 days.")
            return
        
        # Extract unique states
        df_last_7_days_states = df_last_7_days["UsState"].unique()

        log.info(f"Temporary downloading random photos from each state that uploaded in {self.cfg.temp.temp_image_dir}")

        for state in df_last_7_days_states:
            state_df = df_last_7_days[df_last_7_days["UsState"] == state]
            jpg_df = state_df[state_df['ImageURL'].str.endswith('.JPG')]

            if jpg_df.empty:
                log.info(f"No .JPG images found for state: {state}")
                continue

            num_images = min(len(jpg_df), self.num_images_to_inspect) # Select random images
            log.info(f"Selecting {num_images} images for state: {state}")

            random_imageurls = jpg_df['ImageURL'].sample(n=num_images).tolist()

            for url in random_imageurls:
                try:
                    download_from_url(url, self.cfg.temp.temp_image_dir)
                except Exception as e:
                    log.error(f"Failed to download image from {url}: {e}")        

    def plotting_sample_images_and_exif(self) -> None:    
        """ Plots sample images along with important EXIF information."""
        log.info(f"Plotting images with exif data of images selected for sampling")
        df = self.df.copy()
        # remove existing random samples if generating samples again
        try:
            [os.remove(os.path.join(self.inspect_dir, file_name)) for file_name in os.listdir(self.inspect_dir)] 
        except OSError as e:
            log.error(f"Error while removing existing files: {e}")

        temp_image_dir = self.cfg.temp.temp_image_dir

        for filename in os.listdir(temp_image_dir):
            if filename.endswith(('.JPG','.jpg', '.jpeg', '.png', '.gif')):  # Filter for image file extensions
                image_path=(os.path.join(temp_image_dir, filename))
                image_name = os.path.basename(image_path)
            else:
                log.info("Error: check the files in {self.cfg.temp.temp_image_dir}.")

            try:
                exif_info = get_exif_data(image_path)
                # Select EXIF values of interest 
                selected_tags = ['Image DateTime', 'EXIF ExposureTime', 'EXIF ISOSpeedRatings', 'EXIF FNumber', 'EXIF FocalLength']
                selected_info = {tag: value for tag, value in exif_info.items() if tag in selected_tags}
                 
                # Add additional information from df for the same image
                selected_info['Username'] = df.loc[df['Name']==image_name, 'Username'].iloc[0]
                selected_info['Species'] = df.loc[df['Name']==image_name, 'Species'].iloc[0]
                selected_info['UploadDateTimeUTC'] = df.loc[df['Name']==image_name, 'UploadDateTimeUTC'].iloc[0]
                selected_info['HasMatchingJpgAndRaw'] = df.loc[df['Name']==image_name, 'HasMatchingJpgAndRaw'].iloc[0]

                # Plot the image
                image = Image.open(image_path)

                fig, (ax_image, ax_info) = plt.subplots(1, 2, figsize=(8, 3))  
                ax_image.imshow(image)
                ax_image.axis('off') 
                ax_image.set_title('Sample image with EXIF Data')

                # Annotate the plot with selected EXIF information
                exif_text = '\n'.join([f"{tag} : {value}" for tag, value in selected_info.items()])
                ax_info.text(0, 1, exif_text, fontsize=10, color='black', verticalalignment='top')
                ax_info.axis('off')

                # save the image with selected exif data
                plt.savefig(self.inspect_dir/os.path.basename(image_path), dpi=200) # dpi=300 for clear images
                plt.clf() # Clear the plot for the next image
                plt.close(fig)

                os.remove(image_path) # remove the temp image after plotting
            except Warning as e:
                log.error(f"Error plotting images for inspection {image_path}: {e}")

def main(cfg: DictConfig) -> None:
    """Main function to execute batch report tasks."""
    log.info(f"Starting {cfg.general.task}")
    imginspect = InsepctRecentUploads(cfg)
    imginspect.download_images_temp()
    imginspect.plotting_sample_images_and_exif()
    log.info(f"{cfg.general.task} completed.")