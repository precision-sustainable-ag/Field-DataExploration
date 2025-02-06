import os
import cv2
import logging
import shutil

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
        self.csv_path = find_most_recent_data_csv(cfg.paths.datadir)
        self.cfg = cfg

        self.df = self.read()
        self.config_inspection_dir()

        self.temp_image_dir = Path(cfg.paths.temp_image_dir)
        self.temp_image_dir.mkdir(exist_ok=True, parents=True)

        self.num_past_days_to_inspect = cfg.inspection.num_past_days_to_inspect
        self.num_images_to_inspect = cfg.inspection.num_images_to_inspect

    def read(self) -> pd.DataFrame:
        """Read and load data from a CSV file."""
        log.info(f"Reading path: {self.csv_path}")
        df = pd.read_csv(self.csv_path, dtype={"SubBatchIndex": str}, low_memory=False)
        return df

    def config_inspection_dir(self) -> None:
        """Create directory for inspecting samples."""
        log.info(f"Creating output path for the results")
        self.report_dir = Path(self.cfg.paths.missing_batch_folders).parent
        self.report_dir.mkdir(exist_ok=True, parents=True)

        self.inspect_dir = Path(self.cfg.paths.inspectdir)
        self.inspect_dir.mkdir(exist_ok=True, parents=True)

    def download_images_temp(self) -> None:
        """Download up to selected random images from each state that uploaded for last __ selected days."""
        
        # Extract the date from UploadDateTimeUTC
        self.df['upload_date'] = pd.to_datetime(self.df['UploadDateTimeUTC'].str[:10], format='%Y-%m-%d')

        # Define the date range for the last __ selected days
        current_date_time = pd.to_datetime(datetime.now().date())
        targeted_days_ago = current_date_time - timedelta(self.num_past_days_to_inspect)

        # Filter the DataFrame for the last 7 days
        df_targeted_days = self.df[(self.df['upload_date'] >= targeted_days_ago) & (self.df['upload_date'] <= current_date_time)]
        
        if df_targeted_days.empty:
            log.info(f"No uploads in the last {self.num_past_days_to_inspect} days.")
            return
        
        # Extract unique states
        df_targeted_days_states = df_targeted_days["UsState"].unique()

        log.info(f"Temporary downloading random photos from each state that uploaded in {self.cfg.paths.temp_image_dir}")

        for state in df_targeted_days_states:
            state_df = df_targeted_days[df_targeted_days["UsState"] == state]
            jpg_df = state_df[state_df['ImageURL'].str.endswith('.JPG')]

            if jpg_df.empty:
                log.info(f"No .JPG images found for state: {state}")
                continue

            num_images = min(len(jpg_df), self.num_images_to_inspect) # Selec random images for inspecting
            log.info(f"Selecting {num_images} images for state: {state}")

            random_imageurls = jpg_df['ImageURL'].sample(n=num_images).tolist()

            for url in random_imageurls:
                try:
                    download_from_url(url, self.cfg.paths.temp_image_dir)
                except Exception as e:
                    log.error(f"Failed to download image from {url}: {e}")        

    def plotting_sample_images_and_exif(self) -> None:    
        """ Plots sample images along with important EXIF information."""
        log.info(f"Plotting images with exif data of images selected for sampling")
        try:
            # [os.remove(os.path.join(self.inspect_dir, folder_name)) for folder_name in os.listdir(self.inspect_dir)] 
            [shutil.rmtree(os.path.join(self.inspect_dir, folder_name)) for folder_name in os.listdir(self.inspect_dir)] 
        except OSError as e:
            log.error(f"Error while removing existing directories: {e}")

        temp_image_dir = self.cfg.paths.temp_image_dir

        for filename in os.listdir(temp_image_dir):
            if filename.endswith(('.JPG','.jpg', '.jpeg', '.png', '.gif')):  # Filter for image file extensions
                image_path=(os.path.join(temp_image_dir, filename))
                image_name = os.path.basename(image_path)
            else:
                log.info(f"Error: check the files in {temp_image_dir}.")

            try:
                exif_info = get_exif_data(image_path)
                # Select EXIF values of interest 
                selected_tags = ['Image DateTime', 'EXIF ExposureTime', 'EXIF ISOSpeedRatings', 'EXIF FNumber', 'EXIF FocalLength']
                selected_info = {tag: value for tag, value in exif_info.items() if tag in selected_tags}
                 
                # Add additional information from df for the same image
                selected_info['UsState'] = self.df.loc[self.df['Name']==image_name, 'UsState'].iloc[0]
                selected_info['Username'] = self.df.loc[self.df['Name']==image_name, 'Username'].iloc[0]
                selected_info['Species'] = self.df.loc[self.df['Name']==image_name, 'Species'].iloc[0]
                selected_info['UploadDateTimeUTC'] = self.df.loc[self.df['Name']==image_name, 'UploadDateTimeUTC'].iloc[0]
                selected_info['HasMatchingJpgAndRaw'] = self.df.loc[self.df['Name']==image_name, 'HasMatchingJpgAndRaw'].iloc[0]

                # Plot the image with selected EXIF information
                image = Image.open(image_path)

                fig, (ax_image, ax_info) = plt.subplots(1, 2, figsize=(8, 3))  
                ax_image.imshow(image)
                ax_image.axis('off') 
                ax_image.set_title('Sample image with EXIF Data')

                # Annotate the plot with selected EXIF information
                exif_text = '\n'.join([f"{tag} : {value}" for tag, value in selected_info.items()])
                ax_info.text(0, 1, exif_text, fontsize=10, color='black', verticalalignment='top')
                ax_info.axis('off')

                # create folder with name of state
                state_folder = os.path.join(self.inspect_dir, selected_info['UsState'])
                os.makedirs(state_folder, exist_ok=True)

                # save the image with selected exif data
                plt.savefig(Path(state_folder)/os.path.basename(image_path), dpi=200) # dpi=300 for good quality images
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