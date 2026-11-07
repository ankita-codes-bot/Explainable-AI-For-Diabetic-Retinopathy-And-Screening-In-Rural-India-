import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50, DenseNet121

from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

from src.config import (
    IMAGE_SIZE,
    NUM_CLASSES,
    LEARNING_RATE,
    MODEL_BACKBONE,
    ORDINAL_LOSS_WEIGHT
)
from src.ordinal_loss import ordinal_regression_loss

PREPROCESS_FNS = {
    "EfficientNetB0": effnet_preprocess,
    "ResNet50": resnet_preprocess,
    "DenseNet121": densenet_preprocess,
}

LAST_CONV_LAYER = {
    "EfficientNetB0": "top_conv",
    "ResNet50": "conv5_block3_out",
    "DenseNet121": "relu",
}


def _make_metrics():
    return [
        "accuracy",
        tf.keras.metrics.AUC(name="auc", multi_label=True),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]


def build_model(
    backbone_name=MODEL_BACKBONE,
    input_shape=(*IMAGE_SIZE, 3),
    num_classes=NUM_CLASSES,
    fine_tune_at=None,
    ordinal_weight=None,
):
    """
    DenseNet121 baseline + Ordinal Loss.

    ordinal_weight: optional override for config.ORDINAL_LOSS_WEIGHT.
    Pass this in directly (e.g. 0.30) when you want to sweep the loss
    weight for a quick comparison run WITHOUT editing config.py. Leave
    as None to use the config value, same as before -- this is fully
    backward compatible with existing calls.
    """
    if backbone_name not in PREPROCESS_FNS:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if ordinal_weight is None:
        ordinal_weight = ORDINAL_LOSS_WEIGHT

    if backbone_name == "EfficientNetB0":
        base_model = EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=input_shape, pooling=None
        )
    elif backbone_name == "ResNet50":
        base_model = ResNet50(
            include_top=False, weights="imagenet",
            input_shape=input_shape, pooling=None
        )
    elif backbone_name == "DenseNet121":
        base_model = DenseNet121(
            include_top=False, weights="imagenet",
            input_shape=input_shape, pooling=None
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    last_conv_layer_name = LAST_CONV_LAYER[backbone_name]
    base_model.trainable = False

    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_at:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = layers.Lambda(PREPROCESS_FNS[backbone_name], name="backbone_preprocess")(inputs)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_256"
    )(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"DR_{backbone_name}_{num_classes}class")
    loss_fn = ordinal_regression_loss(
        num_classes=num_classes,
        weight=ordinal_weight,
        label_smoothing=0.1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=_make_metrics()
    )

    model.last_conv_layer_name = last_conv_layer_name
    model.backbone_name = backbone_name
    model.ordinal_weight_used = ordinal_weight

    return model


def unfreeze_for_finetuning(model, fine_tune_at_ratio=0.3, fine_tune_lr=1e-5, ordinal_weight=None):
    """
    Unfreezes the top layers of the backbone for fine-tuning.
    Dynamically extracts num_classes from the model output shape to ensure
    compatibility across both Stage 1 (2 classes) and Stage 2 (4 classes).

    ordinal_weight: optional override, same purpose as in build_model().
    If None, reuses whatever weight the model was originally built with
    (model.ordinal_weight_used, set by build_model above), falling back
    to config.ORDINAL_LOSS_WEIGHT if that attribute isn't present (e.g.
    a model loaded from disk without going through build_model()).
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("Could not find backbone sub-model inside model layers.")

    base_model.trainable = True

    fine_tune_at = int(len(base_model.layers) * fine_tune_at_ratio)

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    num_classes = model.output_shape[-1]

    if ordinal_weight is None:
        ordinal_weight = getattr(model, "ordinal_weight_used", ORDINAL_LOSS_WEIGHT)

    loss_fn = ordinal_regression_loss(
        num_classes=num_classes,
        weight=ordinal_weight,
        label_smoothing=0.1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss=loss_fn,
        metrics=_make_metrics()
    )
    model.ordinal_weight_used = ordinal_weight

    return model