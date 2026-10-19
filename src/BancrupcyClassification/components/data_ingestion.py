import os
from src.BancrupcyClassification.logger import logger
import zipfile
from src.BancrupcyClassification.entity.config_entity import DataIngestionConfig
from dotenv import load_dotenv
load_dotenv()
import subprocess
import kaggle

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
    
    def download_file(self):
        
        cmd = [
            "kaggle", "datasets", "download",
            "-d", self.config.source_PATH,
            "-p", str(self.config.root_dir),
            "--force"
        ]

        if not os.path.exists(self.config.local_data_file):

            subprocess.run(cmd, check=True)
            logger.info(f" Data file download successful")

        else:
            logger.info(f"File already exists")



    def extract_zip_file(self):
        
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)  

        # Ensure the file exists before extraction
        if not os.path.exists(self.config.local_data_file):
            raise FileNotFoundError(f"{self.config.local_data_file} not found. Check Kaggle download filename.")

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
            logger.info("Data file unzip successful")

