import os
import uuid

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to your saved model (using the newer .keras format)
MODEL_PATH = os.path.join(BASE_DIR, "alzheimers_cnn_model.keras")
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "alzheimers_hybrid_model.keras")

# Directory to store generated Grad-CAM visualizations
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Image size used during training
IMG_SIZE = 128

# Class names in the same order as during training
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


def load_model():
    """
    Load and return the trained Keras model.
    Tries to load hybrid model first, falls back to CNN model.
    """
    if os.path.exists(HYBRID_MODEL_PATH):
        model = tf.keras.models.load_model(HYBRID_MODEL_PATH)
        return model, True  # True indicates hybrid model
    else:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, False  # False indicates CNN-only model


model, is_hybrid = load_model()

app = Flask(__name__)


def extract_glcm_features(img_batch):
    """
    Extracts Contrast, Homogeneity, and Entropy for a batch of images.
    Matches the GLCM extraction used in training.
    """
    features = []
    for img in img_batch:
        # Convert to 8-bit gray for GLCM (use first channel if RGB)
        if img.shape[-1] == 3:
            img_gray = img[:, :, 0]  # Use first channel
        else:
            img_gray = img[:, :, 0]
        
        img_uint = (img_gray * 255).astype(np.uint8)
        glcm = graycomatrix(
            img_uint, distances=[5], angles=[0], levels=256, symmetric=True, normed=True
        )
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homo = graycoprops(glcm, 'homogeneity')[0, 0]
        # Entropy calculation
        p_norm = glcm / np.sum(glcm)
        entropy = -np.sum(p_norm * np.log2(p_norm + 1e-10))
        features.append([contrast, homo, entropy])
    return np.array(features, dtype=np.float32)


def preprocess_image(file_storage):
    """
    Preprocess the uploaded image so it matches the training pipeline.

    - Opens with PIL
    - Converts to RGB
    - Resizes to (IMG_SIZE, IMG_SIZE)
    - Converts to numpy array
    - Adds batch dimension: (1, IMG_SIZE, IMG_SIZE, 3)

    Note: The model itself has a Rescaling(1./255) layer as the first layer,
    so we do NOT manually divide by 255 here.
    """
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 128, 128, 3)
    return img_array


def generate_activation_heatmap(img_array, model, is_hybrid_model=False):
    """
    Generate a visually attractive activation map heatmap from convolutional layers.
    This shows which parts of the image activate the model's filters most strongly.
    
    Returns the filename of the saved image, or None on failure.
    """
    try:
        # Find the last convolutional layer (usually has the most abstract features)
        last_conv_layer = None
        
        # Try to find the last Conv2D layer
        for i in range(len(model.layers) - 1, -1, -1):
            layer = model.layers[i]
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            # Fallback: find any Conv2D layer
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer
                    break
        
        if last_conv_layer is None:
            return None
        
        # Create a model that outputs the activation maps
        activation_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=last_conv_layer.output
        )
        
        # Get activations
        if is_hybrid_model:
            img_for_glcm = img_array / 255.0
            texture_features = extract_glcm_features(img_for_glcm)
            texture_features = np.expand_dims(texture_features, axis=0)
            activations = activation_model([img_array, texture_features])
        else:
            activations = activation_model(img_array)
        
        # Get the first image's activations
        act = activations[0].numpy()  # Shape: (H, W, num_filters)
        
        # Average across all filters to get a single heatmap
        heatmap = np.mean(act, axis=-1)
        
        # Normalize to [0, 1]
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply a vibrant colormap (VIRIDIS or PLASMA for better visibility)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_PLASMA)
        
        # Get original image
        img_original = img_array[0] / 255.0
        img_original_uint8 = (img_original * 255).astype(np.uint8)
        
        # Convert to grayscale for overlay
        if img_original_uint8.shape[-1] == 3:
            img_gray = cv2.cvtColor(img_original_uint8, cv2.COLOR_RGB2GRAY)
            img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        else:
            img_gray_rgb = img_original_uint8
        
        # Create a more visually appealing overlay
        # Use higher opacity for the heatmap to make it more vibrant
        superimposed = cv2.addWeighted(heatmap_colored, 0.6, img_gray_rgb, 0.4, 0)
        
        # Save the image
        filename = f"heatmap_{uuid.uuid4().hex}.png"
        save_path = os.path.join(STATIC_DIR, filename)
        
        # Create a nice visualization with side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(img_original)
        axes[0].set_title('Original MRI', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap overlay
        axes[1].imshow(superimposed)
        axes[1].set_title('Activation Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2, dpi=120)
        plt.close()
        
        return filename
    except Exception as e:
        print(f"Activation heatmap error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_saliency_heatmap(img_array, model, pred_idx, is_hybrid_model=False):
    """
    Generate a class-specific gradient saliency heatmap (why the model predicted this class).

    Returns the filename of the saved image, or None on failure.
    """
    try:
        # Prepare inputs as tensors to compute gradients
        img_var = tf.convert_to_tensor(img_array)

        with tf.GradientTape() as tape:
            tape.watch(img_var)

            if is_hybrid_model:
                img_for_glcm = img_array / 255.0
                texture_features = extract_glcm_features(img_for_glcm)
                texture_features = np.expand_dims(texture_features, axis=0)
                preds = model([img_var, texture_features], training=False)
            else:
                preds = model(img_var, training=False)

            class_channel = preds[:, pred_idx]

        # Gradients w.r.t. input image
        grads = tape.gradient(class_channel, img_var)
        if grads is None:
            return None

        # Take absolute max across color channels to get a 2D saliency map
        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]

        # Normalize to [0, 1]
        saliency -= tf.reduce_min(saliency)
        max_val = tf.reduce_max(saliency)
        if max_val > 0:
            saliency = saliency / max_val
        saliency = saliency.numpy()

        # Resize to IMG_SIZE
        saliency_resized = cv2.resize(saliency, (IMG_SIZE, IMG_SIZE))
        saliency_uint8 = np.uint8(255 * saliency_resized)

        # Apply a high-contrast colormap for saliency (Turbo)
        saliency_colored = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_TURBO)

        # Original image (0-255 uint8)
        img_original = img_array[0]
        img_original_uint8 = img_original.astype(np.uint8)

        # Grayscale for overlay
        if img_original_uint8.shape[-1] == 3:
            img_gray = cv2.cvtColor(img_original_uint8, cv2.COLOR_RGB2GRAY)
            img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        else:
            img_gray_rgb = img_original_uint8

        # Overlay with higher heatmap opacity for clarity
        overlay = cv2.addWeighted(saliency_colored, 0.65, img_gray_rgb, 0.35, 0)

        # Save side-by-side (original + saliency overlay)
        filename = f"saliency_{uuid.uuid4().hex}.png"
        save_path = os.path.join(STATIC_DIR, filename)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_original_uint8.astype(np.uint8))
        axes[0].set_title("Original MRI", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Saliency Heatmap (class-specific)", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.2, dpi=120)
        plt.close()

        return filename
    except Exception as e:
        print(f"Saliency heatmap error: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_image(file_storage):
    """
    Run model prediction on the uploaded image and
    return the predicted label and confidence scores.
    Supports both CNN-only and hybrid (CNN + GLCM) models.
    """
    img_array = preprocess_image(file_storage)
    
    if is_hybrid:
        # Hybrid model: needs both image and GLCM texture features
        # For GLCM, we need the image in [0, 1] range
        img_for_glcm = img_array / 255.0
        texture_features = extract_glcm_features(img_for_glcm)
        texture_features = np.expand_dims(texture_features, axis=0)  # (1, 3)
        
        preds = model.predict([img_array, texture_features])
    else:
        # CNN-only model
        preds = model.predict(img_array)

    # Predicted class index
    pred_idx = int(np.argmax(preds[0]))
    pred_label = CLASS_NAMES[pred_idx]

    # Convert probabilities to a dict of {class_name: prob}
    probs = preds[0].tolist()
    class_probs = [
        {"label": CLASS_NAMES[i], "prob": float(probs[i])} for i in range(len(CLASS_NAMES))
    ]

    # Sort by probability (descending) for display
    class_probs.sort(key=lambda x: x["prob"], reverse=True)

    # Also return GLCM features for display
    if is_hybrid:
        img_for_glcm = img_array / 255.0
        texture_features = extract_glcm_features(img_for_glcm)[0]
        glcm_features = {
            "contrast": float(texture_features[0]),
            "homogeneity": float(texture_features[1]),
            "entropy": float(texture_features[2])
        }
    else:
        # Extract GLCM even for CNN model (for display purposes)
        img_for_glcm = img_array / 255.0
        texture_features = extract_glcm_features(img_for_glcm)[0]
        glcm_features = {
            "contrast": float(texture_features[0]),
            "homogeneity": float(texture_features[1]),
            "entropy": float(texture_features[2])
        }

    return pred_label, class_probs, glcm_features, img_array, pred_idx


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    class_probs = None
    glcm_features = None
    heatmap_image_url = None
    saliency_image_url = None
    error = None

    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            error = "Please choose an image file before submitting."
        else:
            file = request.files["image"]
            try:
                prediction, class_probs, glcm_features, img_array, pred_idx = predict_image(file)
                
                # Generate activation heatmap visualization
                heatmap_filename = generate_activation_heatmap(
                    img_array, model, is_hybrid
                )
                if heatmap_filename:
                    heatmap_image_url = url_for("static", filename=heatmap_filename)

                # Generate class-specific saliency heatmap (gradient-based)
                saliency_filename = generate_saliency_heatmap(
                    img_array, model, pred_idx, is_hybrid
                )
                if saliency_filename:
                    saliency_image_url = url_for("static", filename=saliency_filename)
            except Exception as e:
                error = f"Error while processing the image: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        class_probs=class_probs,
        glcm_features=glcm_features,
        heatmap_image_url=heatmap_image_url,
        saliency_image_url=saliency_image_url,
        is_hybrid=is_hybrid,
        error=error,
    )


if __name__ == "__main__":
    # You can change host/port here if needed
    app.run(host="0.0.0.0", port=5001, debug=True)


