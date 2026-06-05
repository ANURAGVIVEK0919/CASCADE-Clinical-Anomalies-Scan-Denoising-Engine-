import numpy as np
import tensorflow as tf
import cv2
import base64
import io
from PIL import Image

def generate_gradcam(model, img_array, class_idx, last_conv_layer_name="post_relu"):
    """
    model       : Loaded ResNet-50V2 functional model
    img_array   : Preprocessed input image array of shape (256, 256, 1), float32 [0.0 - 1.0]
    class_idx   : Predicted organ class index (0: Chest, 1: Brain, 2: Abdomen)
    returns     : Heatmap as uint8 RGB image of shape (256, 256, 3)
    """
    try:
        # Fetch the nested ResNet-50V2 backbone layer
        resnet = model.get_layer("resnet50v2")
        last_conv_layer = resnet.get_layer(last_conv_layer_name)
        
        # Build the Gradient Model
        # Input: main model inputs (256, 256, 1)
        # Outputs: the last convolutional layer of ResNet and the main model's class output
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.get_layer("class_output").output]
        )
        
        # Expand dimensions to batch size (1, 256, 256, 1)
        img_batch = np.expand_dims(img_array, axis=0)
        
        with tf.GradientTape() as tape:
            # Predict outputs
            conv_outputs, predictions = grad_model(img_batch)
            # Fetch prediction loss for target class score
            loss = predictions[:, class_idx]
            
        # Calculate gradients of the target class score w.r.t the feature map outputs
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pool the gradients across spatial dimensions (height, width)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by their pooled gradient importances
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU activation and normalize map between 0.0 and 1.0
        heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        
        # Resize to standard size (256, 256)
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        
        # Scale to standard 8-bit uint8 intensity
        heatmap_scaled = np.uint8(255 * heatmap_resized)
        
        # Apply JET color map (Blue -> Green -> Yellow -> Red) representing class focus
        heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        return heatmap_rgb
        
    except Exception as e:
        print(f"[!] Error generating GradCAM heatmap: {e}")
        # Return a blank/neutral blue mask if GradCAM generation fails
        blank_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        blank_mask[:, :, 2] = 50 # Add faint blue glow
        return blank_mask

def overlay_gradcam(original_img_uint8, heatmap_rgb, alpha=0.45):
    """
    original_img_uint8 : Grayscale original CT slice as RGB uint8 (256, 256, 3)
    heatmap_rgb        : Computed GradCAM heatmap as RGB uint8 (256, 256, 3)
    alpha              : Opacity weight of the heatmap (0.0 to 1.0)
    returns            : Base64 encoded blended PNG string representing Explainable AI overlay
    """
    try:
        # Ensure input images have 3 channels (RGB)
        if original_img_uint8.ndim == 2:
            original_rgb = np.stack([original_img_uint8]*3, axis=-1)
        elif original_img_uint8.ndim == 3 and original_img_uint8.shape[-1] == 1:
            original_rgb = np.squeeze(original_img_uint8, axis=-1)
            original_rgb = np.stack([original_rgb]*3, axis=-1)
        else:
            original_rgb = original_img_uint8
            
        # Perform alpha blending: blended = original * (1 - alpha) + heatmap * alpha
        blended = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0.0)
        
        # Encode to PNG and convert to base64
        pil_img = Image.fromarray(blended)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{encoded_string}"
        
    except Exception as e:
        print(f"[!] Error blending GradCAM overlay: {e}")
        # Graceful fallback to original base64 encoding if blending fails
        if original_img_uint8.ndim == 3 and original_img_uint8.shape[-1] == 1:
            original_img_uint8 = original_img_uint8.squeeze(-1)
        pil_img = Image.fromarray(original_img_uint8)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}"
