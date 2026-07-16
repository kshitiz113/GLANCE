"""
Data Setup — Extract and prepare Fashionpedia images.

Handles:
1. Extracting val_test2020.zip if not already done
2. Creating the expected directory structure
3. Reporting dataset statistics

Usage:
    python setup_data.py
"""

import os
import sys
import zipfile
import shutil
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)


def setup_data():
    """Extract and organize dataset images."""
    logging.basicConfig(level=logging.INFO)

    zip_path = os.path.join(config.PROJECT_ROOT, "val_test2020.zip")
    extract_dir = config.DATA_DIR

    # Check if images already exist
    test_dir = os.path.join(extract_dir, "test")
    if os.path.isdir(test_dir):
        count = len([f for f in os.listdir(test_dir)
                     if os.path.splitext(f)[1].lower() in config.SUPPORTED_EXTENSIONS])
        if count > 0:
            logger.info(f"Dataset already extracted: {count} images in {test_dir}")

            # Also create a symlink/copy to IMAGE_DIR if needed
            if not os.path.isdir(config.IMAGE_DIR) and test_dir != config.IMAGE_DIR:
                os.makedirs(config.IMAGE_DIR, exist_ok=True)
                logger.info(f"Note: Images are in {test_dir}, "
                            f"update config.IMAGE_DIR or use --data-dir flag")
            return test_dir

    # Extract zip
    if os.path.exists(zip_path):
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("Extraction complete")
    else:
        logger.error(f"ZIP file not found: {zip_path}")
        logger.info("Please download val_test2020.zip from:")
        logger.info("  https://github.com/cvdfoundation/fashionpedia")
        sys.exit(1)

    # Verify
    if os.path.isdir(test_dir):
        count = len([f for f in os.listdir(test_dir)
                     if os.path.splitext(f)[1].lower() in config.SUPPORTED_EXTENSIONS])
        logger.info(f"Dataset ready: {count} images in {test_dir}")
        return test_dir
    else:
        logger.error("Extraction failed — no test directory found")
        sys.exit(1)


if __name__ == "__main__":
    setup_data()
