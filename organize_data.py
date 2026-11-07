"""
organize_data.py
-----------------
Optional utility: copies images from the flat data/raw/train/ folder into
class-named subfolders (data/raw/train/No_DR/, .../Mild/, ...) based on
train_labels.csv.

WHY THIS IS OPTIONAL:
This project's training pipeline (build_dataframe_from_csv in
data_loader.py) reads directly from the flat train/ folder + CSV and does
NOT need these subfolders. Run this script only if you specifically want
the class-subfolder structure too -- e.g. to eyeball images by class in
File Explorer, or to use build_dataframe_from_folders() instead.

SAFETY:
- Uses shutil.copy2, never move -- your original flat images and CSV
  are left untouched.
- Skips any file that's already been copied (safe to re-run).
- Does not delete or rename anything.

Run with:
    python organize_data.py
"""

import os
import shutil
import pandas as pd

from src.config import TRAIN_DIR, LABELS_CSV, CLASS_NAMES


def load_labels_csv(csv_path: str = LABELS_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        df = pd.read_csv(csv_path, sep="\t")
    return df


def main():
    df = load_labels_csv()
    print(f"Loaded {len(df)} rows from {LABELS_CSV}")

    # Create class subfolders if they don't exist
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)

    copied = 0
    skipped_already_exists = 0
    skipped_missing_source = 0

    for _, row in df.iterrows():
        id_code = str(row["id_code"])
        diagnosis = int(row["diagnosis"])

        if not (0 <= diagnosis < len(CLASS_NAMES)):
            print(f"⚠ Skipping {id_code}: invalid diagnosis value {diagnosis}")
            continue

        class_name = CLASS_NAMES[diagnosis]
        src_path = os.path.join(TRAIN_DIR, f"{id_code}.png")
        dst_path = os.path.join(TRAIN_DIR, class_name, f"{id_code}.png")

        if not os.path.isfile(src_path):
            skipped_missing_source += 1
            continue

        if os.path.isfile(dst_path):
            skipped_already_exists += 1
            continue

        shutil.copy2(src_path, dst_path)
        copied += 1

    print("\n" + "=" * 50)
    print("ORGANIZE COMPLETE")
    print("=" * 50)
    print(f"Newly copied:            {copied}")
    print(f"Already existed (skip):  {skipped_already_exists}")
    print(f"Source image missing:    {skipped_missing_source}")

    print("\nFinal counts per class subfolder:")
    for class_name in CLASS_NAMES:
        folder = os.path.join(TRAIN_DIR, class_name)
        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        print(f"  {class_name}: {count}")

    print(
        "\nNote: your original flat images in data/raw/train/ and "
        "train_labels.csv were not modified or removed -- this script "
        "only copied files into the new subfolders."
    )


if __name__ == "__main__":
    main()