"""
ordinal_loss.py
----------------
DR severity (0-4) is ORDERED, not just categorical -- predicting
Moderate when the truth is Severe is a smaller mistake than predicting
No_DR when the truth is Severe. Plain categorical cross-entropy treats
every wrong class the same regardless of "distance" from the truth.
This adds a small ordinal penalty on top of standard cross-entropy:
it computes the model's "expected class index" (a soft weighted
average over its softmax probabilities) and penalizes how far that is
from the true class index, via MSE. This directly discourages
far-off mistakes (e.g. Severe predicted as No_DR) more than
close ones (e.g. Severe predicted as Moderate), which is exactly
the kind of error that matters most for a screening tool -- a badly
wrong guess is worse than a slightly-off one.
Returns PER-SAMPLE loss (shape [batch]), so it's compatible with
class_weight in model.fit() -- Keras multiplies per-sample loss by
the sample's class weight before reducing, same as it would with a
built-in loss.
"""
import tensorflow as tf
def ordinal_regression_loss(num_classes: int, weight: float = 0.3, label_smoothing: float = 0.1):
    """
    weight: how much the ordinal penalty contributes relative to
    cross-entropy. 0.0 = behaves exactly like plain cross-entropy.
    Passed in from config.py's ORDINAL_LOSS_WEIGHT so it's tunable
    without editing this file.
    """
    class_indices = tf.constant(list(range(num_classes)), dtype=tf.float32)
    def loss_fn(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            n = tf.cast(tf.shape(y_true_f)[-1], tf.float32)
            y_true_f = y_true_f * (1.0 - label_smoothing) + (label_smoothing / n)
        cce = tf.keras.losses.categorical_crossentropy(y_true_f, y_pred)
        true_class = tf.reduce_sum(y_true_f * class_indices, axis=-1)
        pred_class = tf.reduce_sum(y_pred * class_indices, axis=-1)
        ordinal_penalty = tf.square(true_class - pred_class)
        return cce + weight * ordinal_penalty
    return loss_fn