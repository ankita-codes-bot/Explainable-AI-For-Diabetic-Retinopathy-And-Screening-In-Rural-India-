import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_DIR, LABELS_CSV, IMAGE_SIZE, BATCH_SIZE,
    CLASS_NAMES, VALIDATION_SPLIT, RANDOM_SEED
)


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    if w < 20 or h < 20:
        return image
    return image[y:y + h, x:x + w]


def apply_ben_graham(image: np.ndarray, target_size=IMAGE_SIZE) -> np.ndarray:
    """
    "Ben Graham" preprocessing -- a well-known technique from the
    original Kaggle Diabetic Retinopathy competition. Subtracts a
    heavily-blurred version of the image from itself (weighted), which
    cancels out uneven illumination across the fundus photo and
    sharply boosts LOCAL contrast, making microaneurysms, small
    hemorrhages, and other fine lesions much more visible.

    Expects image already resized to target_size (RGB, float32).
    """
    blur = cv2.GaussianBlur(image, (0, 0), target_size[0] * 0.15)
    image = cv2.addWeighted(image, 4.0, blur, -4.0, 128)
    return np.clip(image, 0, 255).astype(np.float32)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    NOT called by default -- replaced by apply_ben_graham().
    Kept here (unused) in case you want to re-enable for comparison.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def degrade_raw(image: np.ndarray) -> np.ndarray:
    """
    Makes a TRAINING image look more like a low-quality web image:
    random down-scaling, JPEG compression and light blur. Applied to
    the raw image BEFORE Ben Graham, only when training. Validation
    and predict.py never use this.
    """
    h, w = image.shape[:2]
    if np.random.rand() < 0.5:
        s = np.random.uniform(0.3, 0.8)
        small = cv2.resize(
            image, (max(32, int(w * s)), max(32, int(h * s))),
            interpolation=cv2.INTER_AREA
        )
        image = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if np.random.rand() < 0.5:
        q = int(np.random.randint(30, 80))
        ok, enc = cv2.imencode(
            ".jpg", image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q]
        )
        if ok:
            image = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
    if np.random.rand() < 0.3:
        image = cv2.GaussianBlur(
            image, (0, 0), float(np.random.uniform(0.5, 1.5))
        )
    return np.ascontiguousarray(image)


def preprocess_image(image_path: str, target_size=IMAGE_SIZE, degrade=False) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_black_borders(image)
    if degrade:
        image = degrade_raw(image)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = apply_ben_graham(image, target_size)

    image = image.astype(np.float32)
    return image


def build_dataframe_from_folders(train_dir: str = TRAIN_DIR) -> pd.DataFrame:
    records = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = (
            os.path.join(train_dir, class_name)
            if os.path.isdir(os.path.join(train_dir, class_name))
            else os.path.join(train_dir, f"{class_idx}_{class_name}")
        )
        if not os.path.isdir(class_folder):
            print(f"Warning: folder not found for class '{class_name}' -> {class_folder}")
            continue
        for fname in os.listdir(class_folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                records.append({
                    "filepath": os.path.join(class_folder, fname),
                    "label": class_idx,
                    "class_name": class_name
                })
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            "No images found. Check that data/raw/train/<class_folders> "
            "exist and contain .png/.jpg/.jpeg files."
        )
    return df


def build_dataframe_from_csv(csv_path: str = LABELS_CSV, image_dir: str = TRAIN_DIR) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="\t")
    df["filepath"] = df["id_code"].apply(lambda x: os.path.join(image_dir, f"{os.path.splitext(x)[0]}.png"))
    df["label"] = df["diagnosis"].astype(int)
    df["class_name"] = df["label"].apply(lambda i: CLASS_NAMES[i])
    return df[["filepath", "label", "class_name"]]


def get_train_val_split(df):
    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def oversample_minority_classes(df, max_multipliers=None, default_max_multiplier=4):
    """
    Duplicates rows of underrepresented classes so the model sees them
    more often per epoch.

    max_multipliers: optional dict of {label: max_multiplier}, letting
    each class get a different oversampling cap. Any label not present
    falls back to default_max_multiplier.

    IMPORTANT: only call this on train_df, never val_df.
    """
    if max_multipliers is None:
        max_multipliers = {}
    class_counts = df["label"].value_counts()
    max_count = class_counts.max()
    dfs = [df]
    for label in df["label"].unique():
        class_df = df[df["label"] == label]
        count = len(class_df)
        cap = max_multipliers.get(label, default_max_multiplier)
        multiplier = min(cap, max_count // count)
        if multiplier > 1:
            dfs.append(pd.concat([class_df] * (multiplier - 1), ignore_index=True))
    combined = pd.concat(dfs, ignore_index=True)
    return combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def _tf_preprocess_wrapper(filepath, label, num_classes=None, degrade=False):
    if num_classes is None:
        num_classes = len(CLASS_NAMES)

    def _load(path):
        path = path.numpy().decode("utf-8")
        img = preprocess_image(path, degrade=degrade)
        return img.astype(np.float32)
    image = tf.py_function(func=_load, inp=[filepath], Tout=tf.float32)
    image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def augment(image, label):
    """
    Light augmentation (flip + brightness/contrast only), same as the
    version used for the earlier baseline runs.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1 * 255.0)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def build_tf_dataset(df, batch_size=BATCH_SIZE, training=True, num_classes=None):
    filepaths = df["filepath"].values
    labels = df["label"].values
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=RANDOM_SEED)
    dataset = dataset.map(
        lambda fp, lb: _tf_preprocess_wrapper(
            fp, lb, num_classes=num_classes, degrade=training
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    print("Loading dataset index...")
    df = build_dataframe_from_csv()
    print(f"Total images found: {len(df)}")
    print(df["class_name"].value_counts())
    train_df, val_df = get_train_val_split(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    train_ds = build_tf_dataset(train_df, training=True)
    val_ds = build_tf_dataset(val_df, training=False)
    for images, labels in train_ds.take(1):
        print("Batch image shape:", images.shape)
        print("Batch label shape:", labels.shape)
        print("Pixel value range:", float(tf.reduce_min(images)), "-", float(tf.reduce_max(images)))