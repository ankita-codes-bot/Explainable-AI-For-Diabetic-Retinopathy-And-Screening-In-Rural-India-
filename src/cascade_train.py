"""
cascade_train.py
-----------------
Two-stage cascade training.

Stage 1: binary screen (No_DR vs Any_DR), trained on the FULL dataset.
Stage 2: 4-way severity grading (Mild/Moderate/Severe/Proliferative_DR),
         trained ONLY on diseased images.

THIS VERSION (v2) adds:
  - All outputs go to NEW folders (saved_models_v2, logs_v2), so the
    old saved_models and logs folders are never overwritten.
  - mine_and_finetune(): "learning from mistakes". After normal training,
    the images the model still gets wrong (training split only) are shown
    to it more often during a short, low learning rate fine-tune.
  - Evaluation is printed BEFORE and AFTER the mistake-learning step so
    the two can be compared.
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    HEAD_EPOCHS, FINETUNE_EPOCHS, FINETUNE_LEARNING_RATE,
    MODEL_SAVE_DIR, LOG_DIR, CLASS_NAMES, IMAGE_SIZE, MODEL_BACKBONE,
    CLASS_WEIGHT_BOOST_FACTOR, OVERSAMPLE_MAX_MULTIPLIERS,
    ORDINAL_LOSS_WEIGHT, TTA_NUM_AUGMENTS, RANDOM_SEED
)
from src.data_loader import (
    build_dataframe_from_csv, get_train_val_split, build_tf_dataset,
    oversample_minority_classes
)
from src.model import build_model, unfreeze_for_finetuning, LAST_CONV_LAYER
from src.ordinal_loss import ordinal_regression_loss
from tensorflow.keras import layers

# ---- Save everything into NEW folders so old models/logs are safe ----
MODEL_SAVE_DIR = os.path.join(os.path.dirname(MODEL_SAVE_DIR), "saved_models_v2")
LOG_DIR = os.path.join(os.path.dirname(LOG_DIR), "logs_v2")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

STAGE2_CLASS_NAMES = ["Mild", "Moderate", "Severe", "Proliferative_DR"]


STAGE2_CLASS_WEIGHT_BOOST_FACTOR = {
    0: CLASS_WEIGHT_BOOST_FACTOR.get(1, 1.3),  # Mild      (was label 1)
    2: CLASS_WEIGHT_BOOST_FACTOR.get(3, 2.0),  # Severe    (was label 3)
    3: CLASS_WEIGHT_BOOST_FACTOR.get(4, 1.2),  # Proliferative_DR (was label 4)
}


def make_stage1_df(df):
    """Copy of df with 'label' remapped to binary: 0=No_DR, 1=Any_DR."""
    out = df.copy()
    out["label"] = (out["label"] > 0).astype(int)
    return out


def filter_diseased_df(df, label_col="label"):
    """Stage 2 only sees diseased rows (label 1-4)."""
    return df[df[label_col] > 0].copy()


def remap_stage2_labels(df):
    """Remaps label 1-4 -> 0-3 so one-hot encoding works for 4 classes."""
    out = df.copy()
    out["label"] = out["label"] - 1
    return out


def check_stage1_balance(train_bin_df, skew_threshold=0.60):
    counts = train_bin_df["label"].value_counts().to_dict()
    total = sum(counts.values())
    majority_fraction = max(counts.values()) / total if total else 0.0
    is_skewed = majority_fraction > skew_threshold
    return is_skewed, counts, majority_fraction


def get_cascade_callbacks(stage_name: str):
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"best_{stage_name}_cascade_v2.keras")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(LOG_DIR, f"{stage_name}_cascade_v2_history.csv")
        ),
    ]


def train_stage1(train_df, val_df, head_epochs=None, finetune_epochs=None, ordinal_weight=None):
    if head_epochs is None:
        head_epochs = HEAD_EPOCHS
    if finetune_epochs is None:
        finetune_epochs = FINETUNE_EPOCHS

    print("\n=== STAGE 1: Binary screen (No_DR vs Any_DR) ===")
    train_bin = make_stage1_df(train_df)
    val_bin = make_stage1_df(val_df)

    is_skewed, counts, majority_fraction = check_stage1_balance(train_bin)
    print(f"Stage 1 train label counts: {counts} (majority fraction: {majority_fraction:.2%})")

    if is_skewed:
        print(
            f"Stage 1 balance is skewed ({majority_fraction:.1%} majority) -- "
            "applying oversampling to the minority class before training."
        )
        train_bin = oversample_minority_classes(train_bin, max_multipliers={0: 4, 1: 4})
        print("Stage 1 train label counts after oversampling:",
              train_bin["label"].value_counts().to_dict())
    else:
        print("Stage 1 balance is acceptable -- skipping oversampling.")

    train_ds = build_tf_dataset(train_bin, training=True, num_classes=2)
    val_ds = build_tf_dataset(val_bin, training=False, num_classes=2)

    model = build_model(backbone_name=MODEL_BACKBONE, num_classes=2, ordinal_weight=ordinal_weight)
    print(f"Stage 1 (head) trainable params: {model.count_params():,}")

    history_head = model.fit(
        train_ds, validation_data=val_ds, epochs=head_epochs,
        callbacks=get_cascade_callbacks("stage1_head")
    )

    model = unfreeze_for_finetuning(
        model, fine_tune_at_ratio=0.3, fine_tune_lr=FINETUNE_LEARNING_RATE,
        ordinal_weight=ordinal_weight
    )
    print(f"Stage 1 (fine-tune) trainable params: {model.count_params():,}")

    history_finetune = model.fit(
        train_ds, validation_data=val_ds, epochs=finetune_epochs,
        callbacks=get_cascade_callbacks("stage1_finetune")
    )
    return model, (history_head, history_finetune)


def train_stage2(
    train_df, val_df,
    head_epochs=None, finetune_epochs=None, ordinal_weight=None,
    oversample_multipliers=None, class_weight_boost=None,
):
    if head_epochs is None:
        head_epochs = HEAD_EPOCHS
    if finetune_epochs is None:
        finetune_epochs = FINETUNE_EPOCHS
    if oversample_multipliers is None:
        oversample_multipliers = OVERSAMPLE_MAX_MULTIPLIERS
    if class_weight_boost is None:
        class_weight_boost = STAGE2_CLASS_WEIGHT_BOOST_FACTOR

    print("\n=== STAGE 2: Severity grading (Mild/Moderate/Severe/Proliferative_DR) ===")

    train_diseased = filter_diseased_df(train_df)
    val_diseased = filter_diseased_df(val_df)

    train_diseased = oversample_minority_classes(train_diseased, max_multipliers=oversample_multipliers)
    print(f"Stage 2 train set after oversampling: {len(train_diseased)}")
    print(train_diseased["class_name"].value_counts())

    train_diseased = remap_stage2_labels(train_diseased)
    val_diseased_remapped = remap_stage2_labels(val_diseased)

    train_ds = build_tf_dataset(train_diseased, training=True, num_classes=4)
    val_ds = build_tf_dataset(val_diseased_remapped, training=False, num_classes=4)

    class_weight_dict = {i: 1.0 for i in range(4)}
    class_weight_dict.update(class_weight_boost)
    print("Stage 2 class weights:", class_weight_dict)

    model = build_model(backbone_name=MODEL_BACKBONE, num_classes=4, ordinal_weight=ordinal_weight)
    print(f"Stage 2 (head) trainable params: {model.count_params():,}")

    history_head = model.fit(
        train_ds, validation_data=val_ds, epochs=head_epochs,
        class_weight=class_weight_dict,
        callbacks=get_cascade_callbacks("stage2_head")
    )

    model = unfreeze_for_finetuning(
        model, fine_tune_at_ratio=0.3, fine_tune_lr=FINETUNE_LEARNING_RATE,
        ordinal_weight=ordinal_weight
    )
    print(f"Stage 2 (fine-tune) trainable params: {model.count_params():,}")

    history_finetune = model.fit(
        train_ds, validation_data=val_ds, epochs=finetune_epochs,
        class_weight=class_weight_dict,
        callbacks=get_cascade_callbacks("stage2_finetune")
    )
    return model, (history_head, history_finetune), val_diseased


def _tta_predict(model, images, num_augments=TTA_NUM_AUGMENTS):
    views = [images]
    views.append(tf.image.flip_left_right(images))
    views.append(tf.image.flip_up_down(images))
    views.append(tf.image.rot90(images, k=2))
    views = views[:num_augments]

    batch_probs = None
    for view in views:
        preds = model.predict(view, verbose=0)
        batch_probs = preds if batch_probs is None else batch_probs + preds
    return batch_probs / len(views)


def evaluate_cascade(stage1_model, stage2_model, val_df, stage1_threshold=0.35, use_tta=False):
    """Evaluates the cascade on the ORIGINAL 5-class val_df."""
    print(f"\n=== CASCADE EVALUATION (stage1_threshold={stage1_threshold}, TTA={use_tta}) ===")

    full_val_ds = build_tf_dataset(val_df, training=False, num_classes=5)
    y_true, y_pred = [], []

    for images, one_hot in full_val_ds:
        true_idx = np.argmax(one_hot.numpy(), axis=1)

        if use_tta:
            s1_softmax = _tta_predict(stage1_model, images)
        else:
            s1_softmax = stage1_model.predict(images, verbose=0)
        s1_probs = s1_softmax[:, 1]
        any_dr = s1_probs >= stage1_threshold

        batch_pred = np.zeros(len(true_idx), dtype=int)
        if any_dr.any():
            diseased_images = images.numpy()[any_dr] if not use_tta else tf.boolean_mask(images, any_dr)
            if use_tta:
                s2_probs = _tta_predict(stage2_model, diseased_images)
            else:
                s2_probs = stage2_model.predict(diseased_images, verbose=0)
            s2_pred = np.argmax(s2_probs, axis=1) + 1
            batch_pred[any_dr] = s2_pred

        y_true.extend(true_idx)
        y_pred.extend(batch_pred)

    print("\nCascade Classification Report (5-class, combined):")
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print(report_str)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    print("\nCascade Confusion Matrix (rows=true, cols=predicted):")
    header = "              " + "".join(f"{name[:10]:>12}" for name in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i][:12]:<14}" + "".join(f"{v:>12}" for v in row))

    return y_true, y_pred, cm


def mine_and_finetune(model, train_df_l, val_df_l, num_classes, stage_name,
                      epochs=3, lr=1e-5, repeat=2, ordinal_weight=None):
    """
    Learning from mistakes: find TRAINING images the model gets wrong,
    show them to the model more often, and fine-tune briefly at a low
    learning rate. Uses training images only, never validation images.
    """
    eval_ds = build_tf_dataset(train_df_l, training=False, num_classes=num_classes)
    pred = np.argmax(model.predict(eval_ds, verbose=0), axis=1)
    wrong = train_df_l[pred != train_df_l["label"].values]
    print(f"{stage_name}: {len(wrong)} of {len(train_df_l)} training images misclassified")
    if len(wrong) == 0:
        return model

    hard_df = pd.concat([train_df_l] + [wrong] * repeat, ignore_index=True)
    hard_df = hard_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_ds = build_tf_dataset(hard_df, training=True, num_classes=num_classes)
    val_ds = build_tf_dataset(val_df_l, training=False, num_classes=num_classes)

    model = unfreeze_for_finetuning(
        model, fine_tune_at_ratio=0.3, fine_tune_lr=lr, ordinal_weight=ordinal_weight
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=get_cascade_callbacks(stage_name)
    )
    return model


def cascade_train_pipeline(
    head_epochs=None, finetune_epochs=None, ordinal_weight=None,
    oversample_multipliers=None, class_weight_boost=None,
    save_suffix="", mine_epochs=3,
):
    print("Loading dataset...")
    df = build_dataframe_from_csv()
    train_df, val_df = get_train_val_split(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    stage1_model, _ = train_stage1(
        train_df, val_df, head_epochs=head_epochs,
        finetune_epochs=finetune_epochs, ordinal_weight=ordinal_weight,
    )
    stage2_model, _, val_diseased = train_stage2(
        train_df, val_df, head_epochs=head_epochs,
        finetune_epochs=finetune_epochs, ordinal_weight=ordinal_weight,
        oversample_multipliers=oversample_multipliers,
        class_weight_boost=class_weight_boost,
    )

    stage1_model.save(os.path.join(MODEL_SAVE_DIR, f"cascade_stage1_final{save_suffix}.keras"))
    stage2_model.save(os.path.join(MODEL_SAVE_DIR, f"cascade_stage2_final{save_suffix}.keras"))
    print("\n##### BEFORE learning from mistakes #####")
    evaluate_cascade(stage1_model, stage2_model, val_df, use_tta=False)

    # Learning from mistakes (training images only, never validation)
    stage1_model = mine_and_finetune(
        stage1_model, make_stage1_df(train_df), make_stage1_df(val_df),
        2, "stage1_mined", epochs=mine_epochs, ordinal_weight=ordinal_weight)
    stage2_model = mine_and_finetune(
        stage2_model, remap_stage2_labels(filter_diseased_df(train_df)),
        remap_stage2_labels(val_diseased),
        4, "stage2_mined", epochs=mine_epochs, ordinal_weight=ordinal_weight)

    stage1_model.save(os.path.join(MODEL_SAVE_DIR, f"cascade_stage1_final{save_suffix}_mined.keras"))
    stage2_model.save(os.path.join(MODEL_SAVE_DIR, f"cascade_stage2_final{save_suffix}_mined.keras"))
    print("\n##### AFTER learning from mistakes #####")
    evaluate_cascade(stage1_model, stage2_model, val_df, use_tta=False)
    evaluate_cascade(stage1_model, stage2_model, val_df, use_tta=True)
    return stage1_model, stage2_model


if __name__ == "__main__":
    cascade_train_pipeline(save_suffix="_v2")