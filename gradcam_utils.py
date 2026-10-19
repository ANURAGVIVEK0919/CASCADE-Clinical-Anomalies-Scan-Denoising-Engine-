import numpy as np
import tensorflow as tf
import cv2
import base64
import io
from PIL import Image

def generate_gradcam(model, img_array, class_idx=None, last_conv_layer_name="post_relu"):
    """
    Generate a Gradient-weighted Class Activation Mapping (Grad-CAM) heatmap.

    model               : Loaded multi-task Diagnostic Model (e.g., Diagnostic_ResNet50V2)
    img_array           : Input image array of shape (H, W), (H, W, 1), or (1, H, W, 1)
    class_idx           : Target class index to explain (0: Chest, 1: Brain, 2: Abdomen).
                          If None or invalid, defaults to winning class prediction.
    last_conv_layer_name: Name of target conv layer in backbone (default: 'post_relu')
    returns             : RGB heatmap as uint8 array (H, W, 3), or None if generation fails.
    """
    if model is None or img_array is None:
        return None

    try:
        # 1. Dynamically identify backbone / feature extraction layer
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) or 'resnet' in layer.name.lower():
                base_model = layer
                break
        if base_model is None:
            base_model = model

        # 2. Dynamically locate target conv layer inside backbone
        target_conv_layer = None
        if last_conv_layer_name and hasattr(base_model, 'get_layer'):
            try:
                target_conv_layer = base_model.get_layer(last_conv_layer_name)
            except (ValueError, KeyError):
                target_conv_layer = None

        if target_conv_layer is None:
            # Reverse-search for the last convolutional or ReLU activation layer
            for layer in reversed(base_model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower() or 'relu' in layer.name.lower():
                    target_conv_layer = layer
                    break

        if target_conv_layer is None:
            print("[!] Grad-CAM: Could not locate a valid convolutional layer in backbone.")
            return None

        # 3. Dynamically trace classification head layers between backbone and class_output
        head_layers = []
        found_base = False
        for layer in model.layers:
            if layer == base_model:
                found_base = True
                continue
            if found_base:
                if layer.name == 'class_output':
                    head_layers.append(layer)
                    break
                elif layer.name != 'bbox_output':
                    head_layers.append(layer)

        if not head_layers or head_layers[-1].name != 'class_output':
            print("[!] Grad-CAM: Could not trace classification head to 'class_output'.")
            return None

        # 4. Construct feature extractor sub-model from backbone inputs to conv layer
        conv_model = tf.keras.Model(inputs=base_model.inputs, outputs=target_conv_layer.output)

        # 5. Format input image
        if not isinstance(img_array, np.ndarray):
            img_array = np.array(img_array)

        img_float = img_array.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        if img_float.ndim == 2:
            img_tensor = tf.convert_to_tensor(img_float[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
        elif img_float.ndim == 3:
            if img_float.shape[-1] == 1:
                img_tensor = tf.convert_to_tensor(img_float[np.newaxis, :, :, :], dtype=tf.float32)
            else:
                img_tensor = tf.convert_to_tensor(img_float[np.newaxis, :, :, :1], dtype=tf.float32)
        elif img_float.ndim == 4:
            img_tensor = tf.convert_to_tensor(img_float[:, :, :, :1], dtype=tf.float32)
        else:
            print(f"[!] Grad-CAM: Invalid input image dimensions: {img_float.ndim}")
            return None

        # Replicate grayscale to 3 channels if backbone expects 3 channels
        expected_channels = base_model.input_shape[-1] if hasattr(base_model, 'input_shape') and base_model.input_shape else 3
        if expected_channels == 3:
            img_input = tf.concat([img_tensor, img_tensor, img_tensor], axis=-1)
        else:
            img_input = img_tensor

        # 6. Execute forward pass and record gradients with GradientTape
        with tf.GradientTape() as tape:
            conv_outputs = conv_model(img_input)
            tape.watch(conv_outputs)

            x = conv_outputs
            for layer in head_layers:
                x = layer(x, training=False)

            class_preds = x
            num_classes = int(class_preds.shape[-1])
            if class_idx is None or class_idx < 0 or class_idx >= num_classes:
                class_idx = int(tf.argmax(class_preds[0]))

            target_score = class_preds[:, class_idx]

        grads = tape.gradient(target_score, conv_outputs)
        if grads is None:
            print(f"[!] Grad-CAM: Gradients disconnected between class {class_idx} and conv layer.")
            return None

        # 7. Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # 8. Weight feature maps by pooled gradients
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU (only positive contributions to class activation)
        heatmap = tf.maximum(heatmap, 0.0)
        max_heat = tf.math.reduce_max(heatmap)
        if max_heat > 1e-8:
            heatmap = heatmap / max_heat
        else:
            heatmap = tf.zeros_like(heatmap)

        heatmap_np = heatmap.numpy()

        # Resize heatmap to match image spatial dimensions
        target_h = img_array.shape[0] if img_array.ndim in (2, 3) else 256
        target_w = img_array.shape[1] if img_array.ndim in (2, 3) else 256
        heatmap_resized = cv2.resize(heatmap_np, (target_w, target_h))

        # Color map to JET RGB (Blue -> Cyan -> Yellow -> Red)
        heatmap_scaled = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        return heatmap_rgb

    except Exception as e:
        print(f"[!] Grad-CAM computation error: {e}")
        return None

def overlay_gradcam(original_img_uint8, heatmap_rgb, alpha=0.45):
    """
    Overlay Grad-CAM heatmap onto the original CT slice.

    original_img_uint8 : Grayscale or RGB slice as uint8 (H, W) or (H, W, C)
    heatmap_rgb        : Computed GradCAM heatmap as uint8 (H, W, 3), or None
    alpha              : Heatmap blending opacity (0.0 to 1.0)
    returns            : Base64 data URI string ("data:image/png;base64,...") or None if unavailable.
    """
    if heatmap_rgb is None or original_img_uint8 is None:
        return None

    try:
        # Standardize original image to 3-channel RGB uint8
        if original_img_uint8.ndim == 2:
            original_rgb = np.stack([original_img_uint8] * 3, axis=-1)
        elif original_img_uint8.ndim == 3 and original_img_uint8.shape[-1] == 1:
            original_rgb = np.squeeze(original_img_uint8, axis=-1)
            original_rgb = np.stack([original_rgb] * 3, axis=-1)
        else:
            original_rgb = original_img_uint8

        if original_rgb.dtype != np.uint8:
            if original_rgb.max() <= 1.0:
                original_rgb = (original_rgb * 255).astype(np.uint8)
            else:
                original_rgb = original_rgb.astype(np.uint8)

        # Match dimensions if needed
        if original_rgb.shape[:2] != heatmap_rgb.shape[:2]:
            heatmap_rgb = cv2.resize(heatmap_rgb, (original_rgb.shape[1], original_rgb.shape[0]))

        # Alpha blending: blended = original * (1 - alpha) + heatmap * alpha
        blended = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0.0)

        # Encode to PNG and convert to base64
        pil_img = Image.fromarray(blended)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{encoded_string}"

    except Exception as e:
        print(f"[!] Error blending Grad-CAM overlay: {e}")
        return None
