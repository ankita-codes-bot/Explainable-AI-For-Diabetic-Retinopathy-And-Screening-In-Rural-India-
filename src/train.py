"""
train.py
--------
Training pipeline for the DR classification model.

Strategy (2-stage transfer learning):
  Stage 1: Train only the custom head (+ SE attention block) with the
           backbone frozen.
  Stage 2: Unfreeze top backbone layers and fine-tune at a low LR.

Handles:
- Class imbalance via PER-CLASS oversampling (data_loader.py,
  see config.OVERSAMPLE_MAX_MULTIPLIERS) + a LIGHT targeted
  class_weight nudge on Mild/Severe/Proliferative_DR (frequency fix)
  + ordinal loss in model.py (mistake-distance fix). Oversampling now
  does most of the frequency-correction work per class; class_weight
  only fine-tunes on top rather than re-correcting the same imbalance
  a second time -- stacking both at full strength was over-weighting
  Proliferative_DR's effective training signal past the majority
  class (No_DR).
- SE (Squeeze-and-Excitation) attention block in the model, letting
  it learn which backbone features matter most.
- Checkpointing, early stopping, LR reduction on plateau.
- Confusion matrix + full per-class classification report at the end.

CHANGES FROM ORIGINAL VERSION (for consistency with cascade_train.py's
fast-sweep support and the updated model.py):
  - train_pipeline() now accepts optional head_epochs, finetune_epochs,
    and ordinal_weight overrides. None (default) uses config.py's
    values, so existing calls (`train_pipeline()`) behave exactly as
    before -- this is purely additive.
  - build_model() is now called with ordinal_weight=ordinal_weight,
    matching the updated model.py signature.

IMPORTANT (student/research prototype disclaimer):
This model's validation accuracy is a research/learning-project metric,
not a clinical validation result. Real clinical deployment would require
external validation on independent datasets, calibration analysis,
sensitivity/specificity targets set with clinical input, and regulatory
review. Do not present validation accuracy alone as evidence of
clinical safety or accuracy.
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Optional
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    HEAD_EPOCHS, FINETUNE_EPOCHS, FINETUNE_LEARNING_RATE,
    MODEL_SAVE_DIR, MODEL_SAVE_PATH, LOG_DIR,
    MODEL_BACKBONE, CLASS_NAMES,
    CLASS_WEIGHT_BOOST_CLASSES, CLASS_WEIGHT_BOOST_FACTOR,
    OVERSAMPLE_MAX_MULTIPLIERS,
    TTA_NUM_AUGMENTS
)
from src.data_loader import (
    build_dataframe_from_csv, get_train_val_split, build_tf_dataset,
    oversample_minority_classes
)
from src.model import build_model, unfreeze_for_finetuning, LAST_CONV_LAYER


def compute_class_weights(train_df, boost_classes=CLASS_WEIGHT_BOOST_CLASSES, boost_factor=CLASS_WEIGHT_BOOST_FACTOR) -> dict:
    """
    TARGETED class weighting -- boosts Mild/Severe/Proliferative_DR
    (by default) by per-class factors. All other classes stay at
    weight 1.0. Addresses class FREQUENCY imbalance, combined with
    ordinal loss in model.py (which addresses mistake DISTANCE, a
    different problem) -- not redundant with it.

    boost_factor can be either:
      - a dict {class_idx: factor}, giving each boosted class its own
        multiplier (e.g. config.CLASS_WEIGHT_BOOST_FACTOR)
      - a single float, applied uniformly to every class in
        boost_classes (old behavior, still supported)

    Kept deliberately light in the current config -- oversampling in
    data_loader.py now does most of the frequency-correction work per
    class, so this is a fine-tuning nudge on top, not the primary fix.
    """
    class_weight_dict = {i: 1.0 for i in range(len(CLASS_NAMES))}
    for c in boost_classes:
        if isinstance(boost_factor, dict):
            class_weight_dict[c] = boost_factor.get(c, 1.0)
        else:
            class_weight_dict[c] = boost_factor
    print("Targeted class weights:", class_weight_dict)
    return class_weight_dict


def get_callbacks(stage_name: str):
    """Builds standard Keras callbacks for a training stage.
    Uses a unique name so previous stage checkpoints are never overwritten.
    """
    checkpoint_path = os.path.join(
        MODEL_SAVE_DIR,
        f"best_{stage_name}_ordinal_v1.keras"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, f"{stage_name}_ordinal_v1")
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(LOG_DIR, f"{stage_name}_ordinal_v1_history.csv")
        ),
    ]
    return callbacks


def _history_to_native(history_dict):
    """Converts NumPy float32 values in a Keras History.history dict into
    native Python floats, so json.dump() doesn't crash."""
    return {k: [float(v) for v in vals] for k, vals in history_dict.items()}


def train_pipeline(head_epochs=None, finetune_epochs=None, ordinal_weight=None):
    """
    Full end-to-end training pipeline: data loading -> stage 1 (head)
    training -> stage 2 (fine-tuning) -> final save.

    head_epochs / finetune_epochs: override config.HEAD_EPOCHS /
    config.FINETUNE_EPOCHS for this call only. None (default) uses the
    full config values -- existing behavior, unchanged.

    ordinal_weight: override config.ORDINAL_LOSS_WEIGHT for this call
    only (passed through to build_model()/unfreeze_for_finetuning()).
    None (default) uses the config value -- existing behavior, unchanged.
    """
    if head_epochs is None:
        head_epochs = HEAD_EPOCHS
    if finetune_epochs is None:
        finetune_epochs = FINETUNE_EPOCHS

    print("Loading dataset...")
    df = build_dataframe_from_csv()
    print(f"Total dataset size: {len(df)}")
    print(df["class_name"].value_counts())

    train_df, val_df = get_train_val_split(df)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_df = oversample_minority_classes(train_df, max_multipliers=OVERSAMPLE_MAX_MULTIPLIERS)
    print(f"Train after oversampling: {len(train_df)}")
    print(train_df["class_name"].value_counts())

    train_ds = build_tf_dataset(train_df, training=True)
    val_ds = build_tf_dataset(val_df, training=False)

    class_weights = compute_class_weights(train_df)
    print(f"\nStage 1: Training head on top of frozen {MODEL_BACKBONE}...")
    model = build_model(backbone_name=MODEL_BACKBONE, ordinal_weight=ordinal_weight)
    print(f"Total trainable parameters (stage 1): {model.count_params():,}")

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks("stage1_head")
    )
    print("\nStage 2: Fine-tuning top backbone layers...")
    model = unfreeze_for_finetuning(
        model, fine_tune_at_ratio=0.3, fine_tune_lr=FINETUNE_LEARNING_RATE,
        ordinal_weight=ordinal_weight
    )
    print(f"Total trainable parameters (stage 2): {model.count_params():,}")

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks("stage2_finetune")
    )
    model.save(MODEL_SAVE_PATH)
    print(f"\nFinal model saved to: {MODEL_SAVE_PATH}")

    combined_history = {
        "stage1": _history_to_native(history_stage1.history),
        "stage2": _history_to_native(history_stage2.history)
    }
    history_path = os.path.join(LOG_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(combined_history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    best_val_loss = min(history_stage2.history.get("val_loss", [float("inf")]))
    best_val_acc = max(history_stage2.history.get("val_accuracy", [0.0]))
    print(f"Best stage 2 val_loss: {best_val_loss:.4f}")
    print(f"Best stage 2 val_accuracy: {best_val_acc:.4f}")
    evaluate_model(model, val_ds)
    evaluate_model_with_tta(model, val_ds)
    return model
def evaluate_model(model: tf.keras.Model, val_ds: tf.data.Dataset):
    """Prints a final evaluation summary on the validation set:
    Keras built-in metrics, sklearn per-class classification report
    (argmax-based, trustworthy for multi-class precision/recall/F1),
    and a confusion matrix (raw counts + row-normalized percentages)
    so you can see exactly which classes get confused with which --
    e.g. whether Severe mistakes land mostly on Moderate (close, less
    concerning) or No_DR (far, more concerning for a screening tool).
    """
    print("\nFinal Validation Evaluation (Keras built-in metrics):")
    eval_results = model.evaluate(val_ds, verbose=1)
    if not isinstance(eval_results, (list, tuple)):
        eval_results = [eval_results]
    for name, value in zip(model.metrics_names, eval_results):
        print(f"  {name}: {value:.4f}")

    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nPer-Class Classification Report (argmax-based, true numbers):")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    ))

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    print("\nConfusion Matrix (raw counts, rows=true, cols=predicted):")
    header = "              " + "".join(f"{name[:10]:>12}" for name in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>12}" for v in row)
        print(f"{CLASS_NAMES[i][:12]:<14}{row_str}")

    print("\nConfusion Matrix (row-normalized %, rows=true, cols=predicted):")
    print(header)
    for i, row in enumerate(cm_normalized):
        row_str = "".join(f"{v*100:>11.1f}%" for v in row)
        print(f"{CLASS_NAMES[i][:12]:<14}{row_str}")

    cm_path = os.path.join(LOG_DIR, "confusion_matrix.json")
    with open(cm_path, "w") as f:
        json.dump({
            "class_names": CLASS_NAMES,
            "raw_counts": cm.tolist(),
            "row_normalized": cm_normalized.tolist()
        }, f, indent=2)
    print(f"\nConfusion matrix saved to: {cm_path}")

    print(
        "\nNote: precision/recall in Keras built-in metrics above threshold "
        "softmax outputs at 0.5, which under-reports recall for multi-class "
        "problems. The classification_report and confusion matrix above are "
        "the numbers that actually matter for medical evaluation."
    )
    print(
        "This is a student/research prototype. Validation accuracy alone "
        "does not establish clinical accuracy or safety -- real deployment "
        "would require external validation, calibration, and clinical review."
    )


def evaluate_model_with_tta(model: tf.keras.Model, val_ds: tf.data.Dataset, num_augments: int = TTA_NUM_AUGMENTS):
    """
    Test-Time Augmentation (TTA): instead of predicting each validation
    image once, predict it several times through different fixed
    transforms (original, horizontal flip, vertical flip, 180-degree
    rotation) and AVERAGE the softmax probabilities before taking the
    final class. This does not retrain anything -- it only changes how
    the already-trained model is evaluated/used at prediction time.
    """
    print(f"\nRunning Test-Time Augmentation evaluation ({num_augments} views per image)...")

    y_true = []
    all_probs = []

    for images, labels in val_ds:
        batch_true = np.argmax(labels.numpy(), axis=1)
        y_true.extend(batch_true)

        views = [images]
        views.append(tf.image.flip_left_right(images))
        views.append(tf.image.flip_up_down(images))
        views.append(tf.image.rot90(images, k=2))
        views = views[:num_augments]

        batch_probs: Optional[np.ndarray] = None
        for view in views:
            preds = model.predict(view, verbose=0)
            batch_probs = preds if batch_probs is None else batch_probs + preds

        batch_probs = batch_probs / len(views)
        all_probs.append(batch_probs)

    all_probs = np.concatenate(all_probs, axis=0)
    y_pred_tta = np.argmax(all_probs, axis=1)

    print("\nPer-Class Classification Report WITH Test-Time Augmentation:")
    print(classification_report(
        y_true, y_pred_tta,
        target_names=CLASS_NAMES,
        digits=4
    ))

    cm = confusion_matrix(y_true, y_pred_tta)
    print("\nConfusion Matrix WITH TTA (raw counts, rows=true, cols=predicted):")
    header = "              " + "".join(f"{name[:10]:>12}" for name in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>12}" for v in row)
        print(f"{CLASS_NAMES[i][:12]:<14}{row_str}")

    print(
        "\nCompare this report to the single-pass one above -- if "
        "TTA's per-class F1/recall (especially Severe/Proliferative_DR) "
        "is meaningfully better, it's worth keeping TTA active in your "
        "actual deployment/inference code too, not just here."
    )


def load_trained_model(model_path: str = MODEL_SAVE_PATH) -> tf.keras.Model:
    """
    Loads a saved model for inference (used by the backend in Phase 4).
    Uses LAST_CONV_LAYER from model.py -- correct for whichever
    backbone MODEL_BACKBONE points to.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py first."
        )

    model = tf.keras.models.load_model(model_path, compile=False)
    model.last_conv_layer_name = LAST_CONV_LAYER[MODEL_BACKBONE]
    model.backbone_name = MODEL_BACKBONE

    print(f"Loaded model from {model_path} (backbone: {model.backbone_name})")
    return model


if __name__ == "__main__":
    trained_model = train_pipeline()