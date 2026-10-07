import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.
    Since the diagnostic model contains a nested ResNet50V2, we need to extract 
    the gradients from the internal base_model.
    """
    # 1. Identify the internal ResNet50V2 model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'resnet' in layer.name.lower():
            base_model = layer
            break
            
    if base_model is None:
        print("[Grad-CAM] WARNING: Could not find nested ResNet50V2 model.")
        return np.zeros((256, 256), dtype=np.float32)

    last_conv_layer_name = "post_relu" # The final activation in ResNet50V2 before pooling
    
    try:
        # 2. We need to create a new model that maps from the input image 
        # to the activations of the last conv layer AND the final classification output.
        
        # In our architecture, the input goes through: 
        # Input -> Augmentation -> Concatenate -> base_model -> GlobalAveragePooling2D -> Dense -> Class_Output
        
        # Let's build a gradient model dynamically
        # Since we can't easily split the Functional API across nested models, 
        # we compute gradients using a GradientTape by explicitly calling the layers.
        
        with tf.GradientTape() as tape:
            # Reconstruct the forward pass up to the base model
            x = model.layers[0].output # The input layer is already handled by img_array
            
            # Since img_array is already a tensor, we pass it through the model manually 
            # or just use the model's structure.
            
            # The easiest way for nested Functional models is to wrap the specific sub-model
            grad_model = tf.keras.models.Model(
                [base_model.inputs], 
                [base_model.get_layer(last_conv_layer_name).output, base_model.output]
            )
            
            # We need to manually replicate the pre-processing the main model does (Concatenate 3 channels)
            # The input is (1, 256, 256, 1). We need to replicate to 3 channels.
            img_3ch = tf.concat([img_array, img_array, img_array], axis=-1)
            
            # Watch the 3-channel image tensor
            tape.watch(img_3ch)
            
            last_conv_layer_output, base_model_output = grad_model(img_3ch)
            
            # Now we must pass base_model_output through the REST of the main model to get the class prediction!
            # The layers after base_model are: GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, ClassOutput
            x = model.get_layer('global_average_pooling2d')(base_model_output)
            x = model.get_layer('dense')(x)
            x = model.get_layer('batch_normalization')(x)
            
            # Get the actual class output layer (we ignore bbox for GradCAM explanation of class)
            class_output = model.get_layer('class_output')(x)
            
            if pred_index is None:
                pred_index = tf.argmax(class_output[0])
            class_channel = class_output[:, pred_index]

        # 3. Compute gradients of the top class with respect to the feature map
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # 4. Global average pooling of gradients (alpha)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 5. Multiply feature map by pooled gradients
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # 6. Apply ReLU (only positive influences matter)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
        
    except Exception as e:
        print(f"[Grad-CAM] ERROR during heatmap generation: {e}")
        return np.zeros((256, 256), dtype=np.float32)

def overlay_gradcam(img, heatmap, alpha=0.4):
    """
    Overlay the Grad-CAM heatmap on the original image.
    """
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB format
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to RGB if it's grayscale
    if len(img.shape) == 2 or img.shape[2] == 1:
        if len(img.shape) == 3:
            img = np.squeeze(img, axis=-1)
        img = np.uint8(255 * img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = np.uint8(255 * img)
        
    # Superimpose the heatmap on the original image
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img
