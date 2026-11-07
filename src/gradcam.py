"""
gradcam.py
----------
Implements Grad-CAM (Gradient-weighted Class Activation Mapping) for
explaining the DR model's predictions.
Why Grad-CAM matters for this project:
In a rural screening context, a black-box "Severe DR" prediction is not
clinically actionable or trustworthy on its own. Grad-CAM produces a
heatmap showing WHICH regions of the retina (e.g., microaneurysms,
hemorrhages, exudates) drove the model's decision -- letting a healthcare
worker visually verify the prediction against the actual image before
referring a patient onward.
Core idea:
1. Run a forward pass, capturing both the last conv layer's feature maps
   AND the final class prediction score.
2. Compute the gradient of the predicted class score w.r.t. those feature
   maps (via GradientTape) -- this tells us "how much does each feature
   map matter for this specific prediction".
3. Global-average-pool those gradients into per-channel importance weights.
4. Weight the feature maps by these importance weights and sum -> a
   coarse localization map highlighting the most influential regions.
5. Apply ReLU (we only care about features that positively influence the
   predicted class) and normalize to [0, 1].
6. Resize the heatmap up to the original image size and overlay it.
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.cm as cm
from src.config import IMAGE_SIZE, CLASS_NAMES
class GradCAM:
    """
    Encapsulates Grad-CAM heatmap generation for a given trained model.
    Usage:
        gradcam = GradCAM(model, last_conv_layer_name=model.last_conv_layer_name)
        heatmap, pred_class, confidence = gradcam.generate_heatmap(preprocessed_image)
        overlay = gradcam.overlay_heatmap(original_image, heatmap)
    """
    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str | None = None):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name or getattr(
            model, "last_conv_layer_name", None
        )
        if self.last_conv_layer_name is None:
            raise ValueError(
                "last_conv_layer_name must be provided or set as a model attribute."
            )
        self.grad_model = self._build_grad_model()
    def _build_grad_model(self) -> tf.keras.Model:
        """
        Constructs a model with two outputs:
          1. The activations of the target conv layer (inside the nested backbone)
          2. The final softmax predictions
        Handles the nested-backbone structure created in model.py, where
        the backbone submodel (EfficientNet/ResNet50/DenseNet121) is found
        by searching for the first nested tf.keras.Model layer, rather than
        assuming a fixed index.
        """
        backbone = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                backbone = layer
                break
        if backbone is None:
            target_layer = self.model.get_layer(self.last_conv_layer_name)
            return tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
        target_conv_layer = backbone.get_layer(self.last_conv_layer_name)
        return tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_conv_layer.output, self.model.output]
        )
    def generate_heatmap(self, image_batch: np.ndarray, class_index: int | None = None):
        """
        Generates a Grad-CAM heatmap for a single preprocessed image.
        Args:
            image_batch: np.ndarray of shape (1, H, W, 3), already preprocessed
                         (same pipeline as data_loader.preprocess_image), raw
                         pixel values in [0, 255] float32 -- the model's own
                         Lambda(backbone_preprocess) layer applies the
                         backbone-specific transform internally, so this
                         input should NOT be normalized to [0,1] beforehand.
            class_index: which class to explain. If None, uses the model's
                         top predicted class (standard "explain the prediction" mode).
        Returns:
            heatmap: np.ndarray (h, w) normalized to [0, 1] -- raw, small-resolution
            predicted_class_index: int
            confidence: float (softmax probability of the explained class)
            all_probabilities: np.ndarray of shape (num_classes,)
        """
        image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image_tensor)
            tape.watch(conv_outputs)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            class_score = predictions[:, class_index]
        grads = tape.gradient(class_score, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  
        conv_outputs = conv_outputs[0]  
        heatmap = conv_outputs @ tf.expand_dims(pooled_grads, axis=-1) 
        heatmap = tf.squeeze(heatmap)  
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        confidence = float(predictions[0][class_index])
        all_probabilities = predictions[0].numpy()
        return heatmap.numpy(), class_index, confidence, all_probabilities
    def overlay_heatmap(self, original_image: np.ndarray, heatmap: np.ndarray,
                         alpha: float = 0.4, colormap=None) -> np.ndarray:
        """
        Overlays the Grad-CAM heatmap on top of the original (unprocessed,
        display-resolution) retinal image.
        Args:
            original_image: np.ndarray (H, W, 3), RGB, uint8, values [0, 255]
            heatmap: raw heatmap from generate_heatmap(), shape (h, w), values [0, 1]
            alpha: blend strength of heatmap over the original image
            colormap: matplotlib colormap used to color the heatmap ('jet' is
                      standard for medical imaging -- high-activation = red).
                      Defaults to 'jet' if not provided.
        Returns:
            overlay: np.ndarray (H, W, 3), RGB, uint8 -- ready to save/display
        """
        if colormap is None:
            colormap = matplotlib.colormaps["jet"]
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        colored_heatmap = np.asarray(colormap(heatmap_resized))[:, :, :3]
        colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
        if original_image.dtype != np.uint8:
            original_image = np.clip(original_image, 0, 255).astype(np.uint8)
        colored_heatmap = np.asarray(colored_heatmap, dtype=np.uint8)
        overlay = cv2.addWeighted(original_image, 1 - alpha, colored_heatmap, alpha, 0)
        return overlay
    def explain(self, preprocessed_image: np.ndarray, original_image: np.ndarray,
                class_index: int | None = None):
        """
        Convenience method combining heatmap generation + overlay in one call.
        This is the primary method the Flask/FastAPI backend (Phase 4) will use.
        Args:
            preprocessed_image: (H, W, 3) float32, raw pixel values in [0, 255]
                                 -- output of data_loader.preprocess_image().
                                 Do NOT normalize this to [0,1]; the model's
                                 internal Lambda layer expects raw pixels.
            original_image: (H, W, 3) uint8 [0,255] -- for a clean visual overlay
            class_index: optional, explain a specific class instead of top prediction
        Returns:
            dict with prediction results + heatmap overlay image
        """
        image_batch = np.expand_dims(preprocessed_image, axis=0)
        heatmap, pred_idx, confidence, all_probs = self.generate_heatmap(
            image_batch, class_index=class_index
        )
        overlay = self.overlay_heatmap(original_image, heatmap)
        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "predicted_class_index": pred_idx,
            "confidence": round(confidence * 100, 2),
            "all_class_probabilities": {
                CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(all_probs)
            },
            "heatmap_raw": heatmap,        
            "overlay_image": overlay   
        }
def save_gradcam_result(overlay_image: np.ndarray, output_path: str):
    """Saves an RGB overlay image to disk (converts to BGR for cv2.imwrite)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bgr_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_image)
    print(f"Grad-CAM overlay saved to: {output_path}")
if __name__ == "__main__":
    """
    Quick manual test:
    python -m src.gradcam --image path/to/sample.png
    """
    import argparse
    from src.train import load_trained_model
    from src.data_loader import preprocess_image

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to a retinal image")
    args = parser.parse_args()
    model = load_trained_model()
    preprocessed = preprocess_image(args.image) 
    raw_bgr = cv2.imread(args.image)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    raw_rgb = cv2.resize(raw_rgb, IMAGE_SIZE) 
    gradcam = GradCAM(model)
    result = gradcam.explain(preprocessed, raw_rgb)

    print(f"\nPrediction: {result['predicted_class']} ({result['confidence']}% confidence)")
    print("All class probabilities:")
    for cls, prob in result["all_class_probabilities"].items():
        print(f"  {cls}: {prob}%")

    save_gradcam_result(
        result["overlay_image"],
        os.path.join("outputs", "gradcam_results", "sample_overlay.png")
    )