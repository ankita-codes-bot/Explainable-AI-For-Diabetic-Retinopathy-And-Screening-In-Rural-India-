"""
merge_ddr.py
-------------
REVISED: only merges DDR's Proliferative_DR images (diagnosis=4) --
NOT Mild, Moderate, or Severe.

WHY: an earlier version merged all 5 DDR classes. Result: Proliferative_DR
F1 improved (0.589 -> 0.739), but Mild collapsed (F1 0.621 -> 0.351,
confusion matrix showed a near-random 3-way split between No_DR/Mild/
Moderate), Severe got worse (F1 0.554 -> 0.439), and even No_DR's
precision dropped (96% -> 87%). Diagnosis: DDR's graders and Aptos/
IDRiD's graders very likely draw the Mild/No_DR and Mild/Moderate
boundaries slightly differently -- this is the single most
inter-grader-inconsistent boundary in DR grading generally. Merging
introduces CONTRADICTORY examples of "Mild", not more signal, and
oversampling then amplifies that contradiction.

Only Proliferative_DR showed a clean, consistent improvement, so this
version merges ONLY diagnosis=4 rows -- keeping Mild/Moderate/Severe
sourced purely from Aptos+IDRiD, where they were already working.

Run merge_idrid.py FIRST, then this script.
"""

import os
import cv2
import pandas as pd

PROJECT_ROOT = r"C:\Users\nikig\dr_screening_model"
DDR_LABELS_CSV = r"D:\Downloads\archive (1)\DR_grading.csv"
DDR_IMAGES_DIR = r"D:\Downloads\archive (1)\DR_grading\DR_grading"

TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "train")
COMBINED_CSV = os.path.join(PROJECT_ROOT, "data", "train_labels_combined.csv")
INCLUDE_ONLY_DIAGNOSIS = [4]


def copy_ddr_images(df):
    print("Converting and copying DDR images...")
    copied, skipped, missing = 0, 0, 0
    for _, row in df.iterrows():
        bare_id = os.path.splitext(row["id_code"])[0]
        src_filename = row["id_code"]
        src_path = os.path.join(DDR_IMAGES_DIR, src_filename)
        dst_path = os.path.join(TRAIN_DIR, f"{bare_id}.png")

        if not os.path.isfile(src_path):
            missing += 1
            continue
        if os.path.isfile(dst_path):
            skipped += 1
            continue

        image = cv2.imread(src_path)
        if image is None:
            print(f"  Warning: could not read {src_filename}, skipping")
            missing += 1
            continue
        cv2.imwrite(dst_path, image)
        copied += 1

    print(f"Converted + copied: {copied}")
    print(f"Already existed (skipped): {skipped}")
    print(f"Missing/unreadable: {missing}")


def build_updated_combined_labels():
    print("Loading DDR labels...")
    ddr_df = pd.read_csv(DDR_LABELS_CSV)
    print(f"DDR total rows: {len(ddr_df)}")
    before = len(ddr_df)
    ddr_df = ddr_df[ddr_df["diagnosis"].isin(INCLUDE_ONLY_DIAGNOSIS)]
    print(f"Filtered to diagnosis {INCLUDE_ONLY_DIAGNOSIS} only: {before} -> {len(ddr_df)} rows")

    print("\nGrade distribution being merged in:")
    print(ddr_df["diagnosis"].value_counts().sort_index())

    copy_ddr_images(ddr_df)
    ddr_df = ddr_df.copy()
    ddr_df["id_code"] = ddr_df["id_code"].apply(lambda x: os.path.splitext(x)[0])
    ddr_df = ddr_df[["id_code", "diagnosis"]]

    print("\nLoading existing combined labels (APTOS + IDRiD)...")
    existing_df = pd.read_csv(COMBINED_CSV, sep="\t")
    print(f"Existing combined rows: {len(existing_df)}")

    updated_df = pd.concat([existing_df, ddr_df], ignore_index=True)
    updated_df.to_csv(COMBINED_CSV, index=False, sep="\t")

    print(f"\nNew combined total rows: {len(updated_df)}")
    print(f"Saved to: {COMBINED_CSV}")


if __name__ == "__main__":
    build_updated_combined_labels()