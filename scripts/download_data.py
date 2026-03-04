"""
Standalone script to download dataset from Google Drive.
Run this before DVC pipeline if network authentication is needed.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --url "YOUR_DRIVE_URL"
"""
import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_URL = "https://drive.google.com/file/d/1mm0L2jRPKqCpyxXRsSJKVoj63-SXncv1/view?usp=sharing"
DEFAULT_OUTPUT = "artifacts/data_ingestion/data.zip"
DEFAULT_UNZIP = "artifacts/data_ingestion/unzipped"


def main():
    parser = argparse.ArgumentParser(description="Download dental OPG dataset")
    parser.add_argument("--url", default=DEFAULT_URL, help="Google Drive URL")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output zip path")
    parser.add_argument("--unzip-dir", default=DEFAULT_UNZIP, help="Extraction directory")
    parser.add_argument("--no-extract", action="store_true", help="Download only, don't extract")
    args = parser.parse_args()

    from dental_opg.entity.config_entity import DataIngestionConfig
    from dental_opg.components.data_ingestion import DataIngestion

    config = DataIngestionConfig(
        root_dir=Path(args.output).parent,
        source_url=args.url,
        local_data_file=Path(args.output),
        unzip_dir=Path(args.unzip_dir),
    )

    di = DataIngestion(config=config)
    di.download_file()

    if not args.no_extract:
        di.extract_zip()
        logger.info(f"Dataset ready at: {args.unzip_dir}")
    else:
        logger.info(f"Dataset downloaded to: {args.output}")


if __name__ == "__main__":
    main()
