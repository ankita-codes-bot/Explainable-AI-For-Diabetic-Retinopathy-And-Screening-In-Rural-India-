"""Lightweight inference entry point for the V3 diabetic retinopathy cascade.

This file does not train or modify any saved checkpoints. It rebuilds the
original DenseNet121 cascade architecture using the same layer structure as the
trained model, then loads the existing saved weights from the .keras bundles.

The original .keras archives use a Lambda layer named "backbone_preprocess" that
contains a serialized DenseNet preprocessing function. That serialized function
is broken on this machine (ValueError: bad marshal data (unknown type code)),
so the inference model here recreates the same preprocessing pipeline using the
canonical DenseNet preprocessing function instead of trying to deserialize the
broken Lambda bytecode.

FIX (see comment at STAGE1_PATH below): stage 1 was pointing at
"cascade_stage1_deploy.keras" -- a different file from the actual trained and
evaluated stage 1 model ("cascade_stage1_final_final_A.keras"). This mismatch
was the root cause of the API always predicting "No_DR" regardless of input:
whatever "cascade_stage1_deploy.keras" contained, it was not the verified
model whose confusion matrix showed correct predictions during evaluation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from src.data_loader import preprocess_image

SAVED_MODELS_DIR = MODEL_DIR / "saved_models"
STAGE1_PATH = SAVED_MODELS_DIR / "cascade_stage1_final_final_A_fixed.keras"
STAGE2_PATH = SAVED_MODELS_DIR / "cascade_stage2_final_final_A_fixed.keras"
STAGE1_THRESHOLD = 0.6
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]


def _extract_model_weights(keras_path: Path) -> str:
    """Extract the raw H5 weights from a .keras bundle without loading the broken Lambda."""
    with zipfile.ZipFile(keras_path, "r") as zf:
        if "model.weights.h5" not in zf.namelist():
            raise FileNotFoundError(f"No model.weights.h5 found inside {keras_path}")
        temp_dir = tempfile.mkdtemp(prefix="dr_cascade_inference_")
        zf.extract("model.weights.h5", temp_dir)
        return os.path.join(temp_dir, "model.weights.h5")


def _build_stage_model(num_classes: int) -> tf.keras.Model:
    """Rebuild the original DenseNet121 cascade head exactly enough to accept the saved weights."""
    inputs = layers.Input(shape=(300, 300, 3), name="input_image")
    x = layers.Lambda(densenet_preprocess, name="backbone_preprocess")(inputs)
    base_model = DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(300, 300, 3),
        pooling=None,
    )
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.BatchNormalization(name="batch_normalization")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense_256",
    )(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"DR_DenseNet121_{num_classes}class")
    return model


def _load_weights_into_model(model: tf.keras.Model, keras_path: Path) -> None:
    """Load saved model weights without deserializing the Keras Lambda bytecode.

    NOTE: the by_name/skip_mismatch fallback below silently leaves any
    non-matching layer at its RANDOM initial weights instead of raising
    an error. That's convenient for minor architecture drift, but it can
    also mask a real problem (e.g. loading weights meant for a different
    architecture) by producing a model that "loads successfully" but is
    partially untrained. The warning print makes this visible instead of
    silent -- if you ever see this warning in production logs, treat the
    model's predictions as suspect until investigated.
    """
    weights_path = _extract_model_weights(keras_path)
    try:
        model.load_weights(weights_path)
    except Exception as e:
        print(
            f"WARNING: primary weight load failed for {keras_path.name} ({e}); "
            "falling back to by_name/skip_mismatch -- some layers may be "
            "randomly initialized. Predictions from this model should be "
            "treated as unreliable until this is investigated."
        )
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)


def _get_stage1_model() -> tf.keras.Model:
    model = _build_stage_model(num_classes=2)
    _load_weights_into_model(model, STAGE1_PATH)
    return model


def _get_stage2_model() -> tf.keras.Model:
    model = _build_stage_model(num_classes=4)
    _load_weights_into_model(model, STAGE2_PATH)
    return model


def _preprocess_image(image_path: str | os.PathLike[str]) -> np.ndarray:
    image = preprocess_image(str(image_path))
    return np.expand_dims(image, axis=0)


def predict_image(image_path) -> Dict[str, object]:
    """Run the V3 two-stage cascade on a single image.

    Returns a dictionary with:
      - predicted_class
      - class_index
      - confidence
      - stage1_probability_any_dr
      - stage2_probabilities (None when stage 2 is not used)
    """
    img = _preprocess_image(image_path)

    stage1_model = _get_stage1_model()
    stage1_logits = stage1_model.predict(img, verbose=0)
    stage1_any_dr_prob = float(stage1_logits[0, 1])

    if stage1_any_dr_prob < STAGE1_THRESHOLD:
        return {
            "predicted_class": "No_DR",
            "class_index": 0,
            "confidence": float(stage1_logits[0, 0]),
            "stage1_probability_any_dr": stage1_any_dr_prob,
            "stage2_probabilities": None,
        }

    stage2_model = _get_stage2_model()
    stage2_logits = stage2_model.predict(img, verbose=0)
    stage2_probs = np.asarray(stage2_logits[0], dtype=np.float32)
    class_index = int(np.argmax(stage2_probs))
    final_class_index = class_index + 1
    confidence = float(stage2_probs[class_index])

    return {
        "predicted_class": CLASS_NAMES[final_class_index],
        "class_index": final_class_index,
        "confidence": confidence,
        "stage1_probability_any_dr": stage1_any_dr_prob,
        "stage2_probabilities": [float(x) for x in stage2_probs.tolist()],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python model/predict.py path/to/image.jpg", file=sys.stderr)
        raise SystemExit(1)

    result = predict_image(sys.argv[1])
    print(json.dumps(result, ensure_ascii=True))