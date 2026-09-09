"""
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability
for the MobileNetV2 transfer-learning disease model.

WHY THIS APPROACH: `build_transfer_model()` (see app/ml/train_transfer.py)
wraps MobileNetV2 as a nested sub-model inside a larger functional model:

    inputs -> preprocess_input(inputs) -> base_model(x) -> GAP -> Dropout
           -> Dense(128, relu) -> Dropout -> Dense(num_classes, softmax)

Naively doing `model.get_layer(<conv_name>).output` on the OUTER model
does not give the tensor produced during the outer model's forward
pass, because `base_model` is called as a single re-usable layer/model,
and Keras only exposes `.output` for a layer's *first* recorded call
node by default. Rather than rely on fragile node-index bookkeeping
(`get_output_at(1)`), we take a more robust and explicit approach:

  1. Build a small "feature extractor" straight from `base_model.input`
     to [<last_conv_layer>.output, base_model.output]. This is
     unambiguous because base_model.input IS the tensor the classifier
     head layers (GAP, Dense, Dense) were literally trained to consume
     (after `mobilenet_v2.preprocess_input`).
  2. Manually replay the classifier head (GAP -> Dense -> Dense) on top
     of `base_model.output` inside a `tf.GradientTape`, watching the
     last convolutional feature map, exactly reproducing the original
     model's math for a specific class logit.
  3. Compute standard Grad-CAM: global-average-pool the gradients per
     channel, weight the conv feature maps by these gradients, ReLU,
     normalize to [0, 1], resize to the image size, and overlay as a
     heatmap on the original image.

This module is only meaningful when a TRAINED model is loaded (see
`app/services/image_model.py`). In fallback-heuristic mode there is no
gradient to compute, so `generate_gradcam` returns `None` — the API and
frontend must handle that gracefully (documented, not hidden).
"""
import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image

from app.ml.labels import IMG_SIZE


class GradCamUnavailableError(Exception):
    """Raised when Grad-CAM cannot be computed for the current model
    (e.g. architecture doesn't match the expected transfer-learning
    structure, or no trained model is loaded)."""
    pass


def _introspect_model(model: tf.keras.Model):
    """Find the nested MobileNetV2 base model and the classifier head
    layers (GAP -> Dense -> Dense) inside the outer functional model
    built by app/ml/train_transfer.py::build_transfer_model.
    Raises GradCamUnavailableError if the expected structure isn't found.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break
    if base_model is None:
        raise GradCamUnavailableError("No nested base (MobileNetV2) sub-model found in the loaded model.")

    gap_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)]
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]

    if not gap_layers or len(dense_layers) < 2:
        raise GradCamUnavailableError(
            "Model architecture does not match the expected GAP -> Dense -> Dense classifier head."
        )

    return base_model, gap_layers[0], dense_layers[0], dense_layers[-1]


def _find_last_conv_layer(base_model: tf.keras.Model) -> str:
    """Return the name of the last layer in base_model whose output is
    a 4D feature map (batch, H, W, C) — this is what Grad-CAM needs."""
    for layer in reversed(base_model.layers):
        try:
            shape = layer.output.shape  # Keras 3: use .output.shape, not the removed .output_shape attribute
        except AttributeError:
            continue
        if shape is not None and len(shape) == 4:
            return layer.name
    raise GradCamUnavailableError("Could not find a 4D convolutional feature map in the base model.")


def compute_gradcam_heatmap(model: tf.keras.Model, preprocessed_input: np.ndarray, class_idx: int) -> np.ndarray:
    """
    Returns a 2D numpy array (H, W) with values in [0, 1] representing
    the Grad-CAM importance heatmap for `class_idx`, at the spatial
    resolution of the last convolutional feature map (before resizing
    to the original image size).
    """
    base_model, gap_layer, dense1, dense_out = _introspect_model(model)
    last_conv_name = _find_last_conv_layer(base_model)

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_name).output, base_model.output],
    )

    x = tf.convert_to_tensor(preprocessed_input, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_output, base_output = feature_extractor(x, training=False)
        tape.watch(conv_output)
        pooled = gap_layer(base_output)
        head = dense1(pooled)
        preds = dense_out(head)
        class_score = preds[:, class_idx]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise GradCamUnavailableError("Gradient computation returned None — check model architecture compatibility.")

    # Global-average-pool the gradients over the spatial dimensions to
    # get one importance weight per channel (the core Grad-CAM step).
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]  # drop batch dim -> (H, W, C)
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)  # ReLU

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def _apply_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Map a [0,1] grayscale heatmap to an RGB 'jet'-like colormap
    without requiring matplotlib/cv2 colormap APIs, for minimal deps."""
    # simple red-yellow-green-blue ramp approximating 'jet'
    r = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def generate_gradcam_overlay_base64(model: tf.keras.Model, image_path: str,
                                     class_idx: int, alpha: float = 0.45) -> str:
    """
    Full pipeline: load the original image, compute the Grad-CAM
    heatmap for `class_idx`, resize it to the image dimensions, overlay
    it on the original image, and return a base64-encoded PNG data URI
    ready to embed directly in the frontend (`<img src="...">`) or a
    PDF report.
    """
    original = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(original, dtype=np.float32)
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(arr.copy())
    preprocessed = np.expand_dims(preprocessed, axis=0)

    heatmap = compute_gradcam_heatmap(model, preprocessed, class_idx)

    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(IMG_SIZE, resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img, dtype=np.float32) / 255.0
    colored_heatmap = _apply_colormap(heatmap_resized)

    original_arr = np.array(original, dtype=np.float32)
    overlay = original_arr * (1 - alpha) + colored_heatmap.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
