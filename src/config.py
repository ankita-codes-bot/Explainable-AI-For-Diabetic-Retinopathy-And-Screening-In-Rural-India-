"""
config.py
---------
Central configuration file for the Diabetic Retinopathy screening project.
Keeping all paths and hyperparameters here avoids hardcoding values
across multiple scripts, and makes the project deployment-ready.
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")
TEST_DIR = os.path.join(RAW_DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "train_labels_combined.csv")

MODEL_SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
for _dir in [PROCESSED_DATA_DIR, MODEL_SAVE_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)
IMAGE_SIZE = (300,300)    
CHANNELS = 3
BATCH_SIZE = 16    
                    
CLASS_NAMES = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative_DR"
]
NUM_CLASSES = len(CLASS_NAMES)
HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 25
EPOCHS = HEAD_EPOCHS + FINETUNE_EPOCHS 

LEARNING_RATE = 1e-4
FINETUNE_LEARNING_RATE = 2e-5  
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42
MODEL_BACKBONE = "DenseNet121"   
MODEL_FILENAME = "dr_model_ordinal_v1.keras"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)

SE_BLOCK_RATIO = 16
ORDINAL_LOSS_WEIGHT = 0.25
OVERSAMPLE_MAX_MULTIPLIERS = {
    1: 4,    
    3: 10, 
    4: 1,   
}
CLASS_WEIGHT_BOOST_CLASSES = ( 1,3)
CLASS_WEIGHT_BOOST_FACTOR = {
    1:1.3, 
    3: 2.0,  
}
TTA_NUM_AUGMENTS = 4