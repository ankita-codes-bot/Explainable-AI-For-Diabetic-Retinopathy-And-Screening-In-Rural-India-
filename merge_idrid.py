"""
One-time script: merges 413 IDRiD training images into your APTOS dataset.
Converts IDRiD .jpg images to .png (so they match your existing pipeline,
no other code changes needed) and builds one new combined labels file.
Your original train_labels.csv is never touched.
"""

import os
import cv2
import pandas as pd

PROJECT_ROOT = r"C:\Users\nikig\dr_screening_model"
APTOS_LABELS_CSV = r"D:\Downloads\aptos2019-blindness-detection\train.csv"
IDRID_LABELS_CSV = r"D:\a. IDRiD_Disease Grading_Training Labels.csv"
IDRID_IMAGES_DIR = r"D:\Downloads\B. Disease Grading\B. Disease Grading\1. Original Images\a. Training Set"

TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "train")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "train_labels_combined.csv")

def convert_and_copy_idrid_images():
    print("Converting and copying IDRiD images...")
    converted = 0
    skipped = 0
    for fname in os.listdir(IDRID_IMAGES_DIR):
        if not fname.lower().endswith(".jpg"):
            continue
        id_code = os.path.splitext(fname)[0]
        src_path = os.path.join(IDRID_IMAGES_DIR, fname)
        dst_path = os.path.join(TRAIN_DIR, f"{id_code}.png")
        if os.path.isfile(dst_path):
            skipped += 1
            continue
        image = cv2.imread(src_path)
        if image is None:
            print(f"  Warning: could not read {fname}, skipping")
            continue
        cv2.imwrite(dst_path, image)
        converted += 1
    print(f"Converted + copied: {converted}")
    print(f"Already existed (skipped): {skipped}")


def build_combined_labels():
    print("\nBuilding combined labels CSV...")
    aptos_df = pd.read_csv(APTOS_LABELS_CSV)[["id_code", "diagnosis"]]
    print(f"APTOS rows: {len(aptos_df)}")
    idrid_df = pd.read_csv(IDRID_LABELS_CSV,sep=None,engine="python")
    idrid_df.columns= [c.strip() for c in idrid_df.columns]
    idrid_df= idrid_df.rename(columns={idrid_df.columns[0]:"id_code",idrid_df.columns[1]:"diagnosis"})
    idrid_df= idrid_df[["id_code","diagnosis"]]
    print(f"IDRiD rows: {len(idrid_df)}")
    combined_df = pd.concat([aptos_df, idrid_df], ignore_index=True)
    combined_df.to_csv(OUTPUT_CSV, index=False, sep="\t")
    print(f"\nCombined total rows: {len(combined_df)}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    convert_and_copy_idrid_images()
    build_combined_labels()