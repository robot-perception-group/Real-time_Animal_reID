"""
Copies images from input folder into an automatically created "renamed_images"
directory, and renames them to RAPID filename convention using metadata from a CSV,
XLS, or XLSX file.
"""

import os
import logging
import pandas as pd
import shutil
import yaml
from pathlib import Path

def load_config(config_path: Path) -> dict:
    """Load YAML config file from .../configs/config_Rename.yaml"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_metadata(meta_path: Path) -> pd.DataFrame:
    """Load the metadata from a CSV or Excel file."""
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    suffix = meta_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(meta_path)
    elif suffix in {".xls", ".xlsx"}:
        df = pd.read_excel(meta_path)
    else:
        raise ValueError(f"Unsupported file format (.csv, .xls, or .xlsx supported)")

    required_columns = ["filename", "img-ID", "animal-ID", "side-info"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Metadata must contains: {required_columns}")

    return df

def rename_and_copy(row: pd.Series, imgs_path: Path, save_dir: Path):
    """Rename and copy a single image based on metadata row"""
    old_filename = str(row["filename"])
    src_image = imgs_path / old_filename

    if not src_image.exists():
        logging.warning(f"Image not found: {src_image}")
        return

    img_ID = str(row["img-ID"]).replace("_", "-")
    animal_ID = str(row["animal-ID"]).replace("_", "-")
    side_info = str(row["side-info"]).replace("_", "-")

    new_filename = f"{animal_ID}_{side_info}_{img_ID}{src_image.suffix}"
    dst_image = save_dir / new_filename

    if dst_image.exists():
        logging.error(f"Destination image already exists: {dst_image}")
        exit()

    shutil.copy2(src_image, dst_image)
    logging.info(f"Renamed {old_filename} --> {new_filename}")

def main():
    """Main function to rename and copy images based on metadata"""

    # LOAD CONFIG
    config_path = (Path(__file__).parent.parent.parent.parent /
                   "config/config_Rename.yaml")
    cfg = load_config(config_path)

    imgs_path = Path(cfg["imgs_path"])
    meta_data_path = Path(cfg["meta_data_path"])

    # SAVE DIR
    save_dir = imgs_path.parent / "renamed_images"
    save_dir.mkdir(parents=True, exist_ok=True)

    # INIT LOGGING
    log_file = save_dir / "rename_images.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Images directory: {imgs_path}")
    logging.info(f"Meta data directory: {meta_data_path}")
    logging.info(f"Save directory: {save_dir}")
    logging.info("----- Starting image renaming process... -----")

    # LOAD METADATA
    df_meta = load_metadata(meta_data_path)

    # ITERATE AND RENAME
    for _, row in df_meta.iterrows():
        rename_and_copy(row, imgs_path, save_dir)

    logging.info("----- Finished image renaming process. -----")


if __name__ == "__main__":
    main()
