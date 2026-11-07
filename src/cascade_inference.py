"""
cascade_inference.py

Combines stage1 (binary screen) + stage2 (severity grading) into one prediction
pipeline, with an explicit, tunable threshold on stage 1 so you control the
sensitivity/specificity trade-off directly, instead of relying on argmax.

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
"""

import numpy as np
import tensorflow as tf

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

def predict_cascade(images: np.ndarray, stage1_model: tf.keras.Model, stage2_model: tf.keras.Model,
                     stage1_threshold: float = 0.6):
    """
    Args:
        images: batch of preprocessed images, shape (N, H, W, 3).
        stage1_model: output of cascade_train.train_stage1 -- a 2-class
            softmax [P(No_DR), P(Any_DR)], matching build_model()'s output
            format (NOT a single sigmoid unit).
        stage2_model: output of cascade_train.train_stage2 -- a 4-class
            softmax over Mild/Moderate/Severe/Proliferative_DR.
        stage1_threshold: probability (on the Any_DR column) above which
            an image is sent to stage 2.
            LOWER this to increase sensitivity (fewer diseased images
            wrongly stopped at "No_DR"), at the cost of more healthy
            images going through stage 2 unnecessarily (stage 2 doesn't
            have a "No_DR" option, so those become false positives at
            some severity). Start around 0.3-0.4 rather than the default
            0.5 -- for screening, false "No_DR" is the error you most
            want to avoid, so bias stage 1 toward flagging borderline
            cases as "Any_DR" and let stage 2 sort them out. Tune this
            against a validation set using the recall you actually want
            on Any_DR (e.g. pick the threshold hitting 98%+ sensitivity),
            not on raw accuracy -- see find_stage1_threshold_for_sensitivity().

    Returns:
        List of dicts, one per image, each with:
            predicted_class (str), predicted_class_idx (int),
            stage1_any_dr_prob (float), stage2_probs (np.ndarray or None)
    """
    stage1_softmax = stage1_model.predict(images, verbose=0)  
    stage1_probs = stage1_softmax[:, 1] 
    any_dr_mask = stage1_probs >= stage1_threshold

    results = [None] * len(images)
    for i in np.where(~any_dr_mask)[0]:
        results[i] = {
            "predicted_class": "No_DR",
            "predicted_class_idx": 0,
            "stage1_any_dr_prob": float(stage1_probs[i]),
            "stage2_probs": None,
        }

    
    diseased_idx = np.where(any_dr_mask)[0]
    if len(diseased_idx) > 0:
        stage2_probs = stage2_model.predict(images[diseased_idx], verbose=0)
        stage2_preds = np.argmax(stage2_probs, axis=1)  
        for j, i in enumerate(diseased_idx):
            class_idx = int(stage2_preds[j]) + 1  
            results[i] = {
                "predicted_class": CLASS_NAMES[class_idx],
                "predicted_class_idx": class_idx,
                "stage1_any_dr_prob": float(stage1_probs[i]),
                "stage2_probs": stage2_probs[j],
            }

    return results


def find_stage1_threshold_for_sensitivity(val_probs: np.ndarray, val_labels: np.ndarray,
                                           target_sensitivity: float = 0.98):
    """
    Sweeps thresholds on a validation set and returns the highest threshold
    that still achieves at least `target_sensitivity` recall on Any_DR (label 1).
    Higher threshold = fewer healthy images unnecessarily forwarded to stage 2,
    while still hitting your sensitivity floor.

    Args:
        val_probs: stage 1 sigmoid outputs on a validation set, shape (N,).
        val_labels: true binary labels (0=No_DR, 1=Any_DR), shape (N,).
        target_sensitivity: minimum acceptable recall on Any_DR.

    Returns:
        (best_threshold: float, achieved_sensitivity: float, achieved_specificity: float)
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best = (0.5, 0.0, 0.0)

    positives = val_labels == 1
    negatives = val_labels == 0

    for t in thresholds:
        preds = val_probs >= t
        sensitivity = preds[positives].sum() / max(positives.sum(), 1)
        specificity = (~preds[negatives]).sum() / max(negatives.sum(), 1)
        if sensitivity >= target_sensitivity and t > best[0]:
            best = (t, sensitivity, specificity)

    return best


def explain_cascade_prediction(stage1_model, stage2_model, preprocessed_image: np.ndarray,
                                original_image: np.ndarray, stage1_threshold: float = 0.35):
    """
    Cascade-aware version of gradcam.GradCAM.explain(). Calling GradCAM.explain()
    DIRECTLY on stage1_model or stage2_model gives WRONG labels: stage1's
    pred_idx is 0/1 (its own binary indexing), stage2's is 0-3 (its own
    4-class indexing), but gradcam.py's explain() looks those indices up in
    the full 5-item CLASS_NAMES list -- e.g. stage2 predicting its own index
    2 ("Severe" in stage2's 0-3 scheme) would silently be labeled
    CLASS_NAMES[2] = "Moderate" instead. This wrapper runs the real cascade
    logic first, THEN builds a GradCAM heatmap from whichever stage actually
    made the final call, with correctly remapped labels.

    Requires gradcam.py's GradCAM class (import it where you call this).

    Returns the same dict shape as GradCAM.explain(), with predicted_class
    correctly reflecting the full 5-class label space.
    """
    from src.gradcam import GradCAM  

    image_batch = np.expand_dims(preprocessed_image, axis=0)
    s1_softmax = stage1_model.predict(image_batch, verbose=0)
    any_dr_prob = float(s1_softmax[0][1])

    if any_dr_prob < stage1_threshold:
        gradcam = GradCAM(stage1_model)
        heatmap, _, confidence, s1_probs = gradcam.generate_heatmap(image_batch, class_index=0)
        overlay = gradcam.overlay_heatmap(original_image, heatmap)
        return {
            "predicted_class": CLASS_NAMES[0],
            "predicted_class_index": 0,
            "confidence": round((1 - any_dr_prob) * 100, 2),
            "all_class_probabilities": {
                "No_DR": round((1 - any_dr_prob) * 100, 2),
                "Any_DR (unresolved by stage 2)": round(any_dr_prob * 100, 2),
            },
            "heatmap_raw": heatmap,
            "overlay_image": overlay,
            "cascade_stage_used": "stage1",
        }
    gradcam = GradCAM(stage2_model)
    s2_class_index = int(tf.argmax(stage2_model(image_batch)[0]).numpy())  # 0-3
    heatmap, _, confidence, s2_probs = gradcam.generate_heatmap(image_batch, class_index=s2_class_index)
    overlay = gradcam.overlay_heatmap(original_image, heatmap)
    real_class_index = s2_class_index + 1 

    return {
        "predicted_class": CLASS_NAMES[real_class_index],
        "predicted_class_index": real_class_index,
        "confidence": round(confidence * 100, 2),
        "all_class_probabilities": {
            "No_DR": round((1 - any_dr_prob) * 100, 2),
            **{CLASS_NAMES[i + 1]: round(float(p) * 100, 2) for i, p in enumerate(s2_probs)}
        },
        "heatmap_raw": heatmap,
        "overlay_image": overlay,
        "cascade_stage_used": "stage2",
    }