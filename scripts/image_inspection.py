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
from utils.utils import find_most_recent_csv, download_from_url, get_exif_data

log = logging.getLogger(__name__)

class InsepctRecentUploads:
    """This class randomly select upto 15 images(less than 15 if the total uploaded images are less),
    plot images and relevant metadata, save plots to reports so they can be inspected."""

    def __init__(self, cfg) -> None:
        """Initialize the InsepctRecentUploads class with confguration and data loading"""
        self.cfg = cfg
        self.csv_path = cfg.data.permanent_table
        self.df = self.read()
        self.config_inspection_dir()

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
        """Download upto 15 random images chosen from last 15 days of uploads for all locations."""
        df = self.df.copy()
        df['capture_date'] = df['BatchID'].str[-10:] #extract last 10 digits (capture date)
        df['capture_date'] = pd.to_datetime(df['capture_date'], format='%Y-%m-%d')

        current_date_time = pd.to_datetime(datetime.now().date())
        fifteen_days_ago = pd.to_datetime(current_date_time - timedelta(days=15)) #calculate date 15 days ago

        df_last_15_days = df[(df['capture_date'] >= fifteen_days_ago) & (df['capture_date'] <= current_date_time)] # df with all columns for last 15 days of uploads
        df_jpg_url = df_last_15_days[df_last_15_days['ImageURL'].str.endswith('.JPG')]

        if len(df_jpg_url) < 15:
            log.warning(f"Less than 15 images available. Randomly selecting {len(df_jpg_url)} images.")
            random_imageurls = df_jpg_url['ImageURL'].sample(n=len(df_jpg_url)).tolist()  # Use all available images
        else:
            random_imageurls = df_jpg_url['ImageURL'].sample(n=15).tolist()  # Randomly select up to 15 images

        random_imageurls = df_jpg_url['ImageURL'].sample(n=min(len(df_jpg_url), 15)).tolist() # generates a list of random URLs upto 15
        
        log.info(f"Temporary downloading random photos in {self.cfg.temp.temp_image_dir}")
        [download_from_url(url, self.cfg.temp.temp_image_dir) for url in random_imageurls] # downloads the images in temp folder
        
    def plotting_sample_images_and_exif(self) -> None:    
        """ Plots sample images along with important EXIF information."""
        log.info(f"Plotting images with exif data of images selected for sampling")

        # remove existing random samples if generating samples again
        try:
            [os.remove(os.path.join(self.inspect_dir, file_name)) for file_name in os.listdir(self.inspect_dir)] 
        except OSError as e:
            log.error(f"Error while removing existing files: {e}")

        temp_image_dir = self.cfg.temp.temp_image_dir

        for filename in os.listdir(temp_image_dir):
            if filename.endswith(('.JPG', '.jpeg', '.png', '.gif')):  # Filter for image file extensions
                image_path=(os.path.join(temp_image_dir, filename))
            else:
                log.info("Error: check the files in {self.cfg.temp.temp_image_dir}.")

            try:
                exif_info = get_exif_data(image_path)
                # Select EXIF values of interest
                selected_tags = ['Image DateTime', 'EXIF ExposureTime', 'EXIF ISOSpeedRatings', 'EXIF FNumber', 'EXIF FocalLength', 'EXIF Flash']
                selected_info = {tag: value for tag, value in exif_info.items() if tag in selected_tags}
                
                # Plot the image
                image = Image.open(image_path)
                plt.imshow(image)
                plt.axis('off')  # Turn off axis
                plt.title('Sample image with EXIF Data')

                # Annotate the plot with selected EXIF information
                exif_text = '\n'.join([f"{tag} : {value}" for tag, value in selected_info.items()])
                plt.annotate(exif_text, xy=(0.5, 0.5), xytext=(10, -10), fontsize=10,
                            textcoords='offset points', ha='left', va='top', color='black',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=(1, 1, 1, 0), edgecolor="black"))

                # save the image with selected exif data
                plt.subplots_adjust(right=1)  # Adjust right margin to make space for annotation
                plt.savefig(self.inspect_dir/os.path.basename(image_path), dpi=300) # dpi=300 for clear images
                os.remove(image_path)

            except Exception as e:
                log.error(f"Error plotting images for inspection {image_path}: {e}")

def main(cfg: DictConfig) -> None:
    """Main function to execute batch report tasks."""
    log.info(f"Starting {cfg.general.task}")
    imginspect = InsepctRecentUploads(cfg)
    imginspect.download_images_temp()
    imginspect.plotting_sample_images_and_exif()
    log.info(f"{cfg.general.task} completed.")