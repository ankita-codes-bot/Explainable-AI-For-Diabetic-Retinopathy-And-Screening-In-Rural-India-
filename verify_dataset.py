"""
verify_dataset.py
------------------
Read-only dataset verification script. Does NOT move, delete, rename,
or modify any image, and does NOT modify train_labels.csv.

Checks:
1. Reads train_labels.csv, auto-detecting comma vs tab separation.
2. Confirms every id_code in the CSV has a matching image file on disk.
3. Confirms every image file in the flat train/ folder has a matching
   CSV row (catches orphan images the CSV doesn't know about).
4. Reports class distribution as declared by the CSV.
5. Checks for duplicate id_code rows in the CSV.
6. Checks for duplicate image filenames on disk.
7. Reports total CSV rows, total image files, and how many overlap.

Run with:
    python -m src.verify_dataset
"""

import os
import pandas as pd

from src.config import TRAIN_DIR, LABELS_CSV, CLASS_NAMES


def load_labels_csv(csv_path: str = LABELS_CSV) -> pd.DataFrame:
    """Tries comma first, falls back to tab if the expected columns
    aren't found -- this project's CSV is tab-separated."""
    df = pd.read_csv(csv_path)

    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        df = pd.read_csv(csv_path, sep="\t")

    if "id_code" not in df.columns or "diagnosis" not in df.columns:
        raise ValueError(
            f"Could not find 'id_code' and 'diagnosis' columns in {csv_path}. "
            f"Found columns: {df.columns.tolist()}"
        )

    return df


def main():
    print("=" * 60)
    print("DATASET VERIFICATION (read-only, nothing will be modified)")
    print("=" * 60)
    df = load_labels_csv()
    print(f"\nCSV loaded: {LABELS_CSV}")
    print(f"Columns detected: {df.columns.tolist()}")
    print(f"Total CSV rows: {len(df)}")
    dup_ids = df[df.duplicated(subset="id_code", keep=False)]
    print(f"\nDuplicate id_code rows in CSV: {len(dup_ids)}")
    if len(dup_ids) > 0:
        print(dup_ids.sort_values("id_code").head(10))
    print("\nClass distribution (from CSV 'diagnosis' column):")
    diagnosis_counts = df["diagnosis"].value_counts().sort_index()
    for label_idx, count in diagnosis_counts.items():
        label_idx = int(str(label_idx))
        class_name= CLASS_NAMES[label_idx] if 0 <= label_idx < len(CLASS_NAMES) else "UNKNOWN"
        print(f"  {label_idx} ({class_name}): {count}")

    invalid_labels = df[~df["diagnosis"].isin(range(len(CLASS_NAMES)))]
    if len(invalid_labels) > 0:
        print(f"\n⚠ WARNING: {len(invalid_labels)} rows have a diagnosis value "
              f"outside the expected range 0-{len(CLASS_NAMES) - 1}:")
        print(invalid_labels.head(10))
    if not os.path.isdir(TRAIN_DIR):
        print(f"\n⚠ ERROR: TRAIN_DIR does not exist: {TRAIN_DIR}")
        return

    disk_files = [
        f for f in os.listdir(TRAIN_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and os.path.isfile(os.path.join(TRAIN_DIR, f))
    ]
    disk_ids = set(os.path.splitext(f)[0] for f in disk_files)

    print(f"\nTotal image files found directly in {TRAIN_DIR}: {len(disk_files)}")

    # Duplicate filenames (same id_code, different extension) on disk
    if len(disk_ids) != len(disk_files):
        print(f"⚠ WARNING: {len(disk_files) - len(disk_ids)} filename collisions "
              f"detected (same id_code with multiple extensions).")
    csv_ids = set(df["id_code"].astype(str))

    in_csv_not_on_disk = csv_ids - disk_ids
    on_disk_not_in_csv = disk_ids - csv_ids
    matched = csv_ids & disk_ids

    print(f"\nMatched (in both CSV and disk): {len(matched)}")
    print(f"In CSV but image MISSING from disk: {len(in_csv_not_on_disk)}")
    if in_csv_not_on_disk:
        sample = list(in_csv_not_on_disk)[:10]
        print(f"  Sample missing id_codes: {sample}")

    print(f"On disk but NOT listed in CSV (orphan images): {len(on_disk_not_in_csv)}")
    if on_disk_not_in_csv:
        sample = list(on_disk_not_in_csv)[:10]
        print(f"  Sample orphan id_codes: {sample}")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total CSV rows:              {len(df)}")
    print(f"Total image files on disk:   {len(disk_files)}")
    print(f"Usable matched pairs:        {len(matched)}")
    print(f"Duplicate CSV id_code rows:  {len(dup_ids)}")
    print(f"Invalid diagnosis values:    {len(invalid_labels)}")
    print(f"Missing images (CSV->disk):  {len(in_csv_not_on_disk)}")
    print(f"Orphan images (disk->CSV):   {len(on_disk_not_in_csv)}")

    if len(matched) == len(df) == len(disk_files) and len(dup_ids) == 0 and len(invalid_labels) == 0:
        print("\n✅ Dataset looks clean: every CSV row has a matching image, "
              "no duplicates, no invalid labels. Safe to proceed to training.")
    else:
        print("\n⚠ Dataset has discrepancies noted above. Review before "
              "relying on the reported counts for training.")


if __name__ == "__main__":
    main()