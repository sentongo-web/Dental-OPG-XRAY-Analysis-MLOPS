import os
import zipfile
import logging
from pathlib import Path
import gdown
import requests
from tqdm import tqdm
from dental_opg.entity.config_entity import DataIngestionConfig

logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _extract_gdrive_id(self, url: str) -> str:
        """Extract Google Drive file ID from URL."""
        if "drive.google.com/file/d/" in url:
            return url.split("/file/d/")[1].split("/")[0]
        elif "id=" in url:
            return url.split("id=")[1].split("&")[0]
        raise ValueError(f"Cannot extract file ID from URL: {url}")

    def download_file(self):
        """Download dataset from Google Drive using gdown."""
        os.makedirs(self.config.root_dir, exist_ok=True)

        if os.path.exists(self.config.local_data_file):
            logger.info(f"Dataset already downloaded: {self.config.local_data_file}")
            return

        logger.info(f"Downloading dataset from: {self.config.source_url}")
        try:
            file_id = self._extract_gdrive_id(self.config.source_url)
            gdown.download(id=file_id, output=str(self.config.local_data_file), quiet=False)
            logger.info(f"Dataset downloaded to: {self.config.local_data_file}")
        except Exception as e:
            logger.error(f"gdown failed: {e}")
            logger.info("Trying direct download fallback...")
            self._fallback_download()

    def _fallback_download(self):
        """Fallback: direct requests download for public Google Drive files."""
        file_id = self._extract_gdrive_id(self.config.source_url)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(download_url, stream=True)

        # Handle large file warning page
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                download_url = f"{download_url}&confirm={value}"
                response = session.get(download_url, stream=True)
                break

        total_size = int(response.headers.get("content-length", 0))
        os.makedirs(self.config.local_data_file.parent, exist_ok=True)

        with open(self.config.local_data_file, "wb") as f, tqdm(
            desc="Downloading", total=total_size, unit="iB", unit_scale=True
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)

        logger.info(f"Downloaded via fallback: {self.config.local_data_file}")

    def extract_zip(self):
        """Extract ZIP archive to unzip directory."""
        if not os.path.exists(self.config.local_data_file):
            raise FileNotFoundError(f"Dataset zip not found: {self.config.local_data_file}")

        os.makedirs(self.config.unzip_dir, exist_ok=True)

        # Check if already extracted
        if any(Path(self.config.unzip_dir).iterdir() if Path(self.config.unzip_dir).exists() else []):
            logger.info(f"Dataset already extracted at: {self.config.unzip_dir}")
            return

        logger.info(f"Extracting dataset to: {self.config.unzip_dir}")
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)

        logger.info(f"Extraction complete. Files: {os.listdir(self.config.unzip_dir)}")
        self._log_dataset_structure()

    def _log_dataset_structure(self):
        """Log the structure of the extracted dataset."""
        unzip_path = Path(self.config.unzip_dir)
        logger.info("Dataset structure:")
        for item in unzip_path.rglob("*"):
            if item.is_dir():
                file_count = len(list(item.glob("*")))
                logger.info(f"  DIR: {item.relative_to(unzip_path)} ({file_count} items)")

        # Count images and labels
        images = list(unzip_path.rglob("*.jpg")) + list(unzip_path.rglob("*.png")) + \
                 list(unzip_path.rglob("*.jpeg")) + list(unzip_path.rglob("*.bmp"))
        labels = list(unzip_path.rglob("*.txt")) + list(unzip_path.rglob("*.xml")) + \
                 list(unzip_path.rglob("*.json"))

        logger.info(f"Total images found: {len(images)}")
        logger.info(f"Total label files found: {len(labels)}")

    def run(self):
        """Execute full data ingestion pipeline."""
        logger.info("=" * 60)
        logger.info("STAGE: Data Ingestion")
        logger.info("=" * 60)
        self.download_file()
        self.extract_zip()
        logger.info("Data Ingestion complete.")
