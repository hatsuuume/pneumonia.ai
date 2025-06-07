import cv2
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load my saved model
model = tf.keras.models.load_model("finalmodel.keras")

# Preprocess uploaded image
def preprocess_image(image):
    image_array = np.array(image).astype(np.float32)     # Convert to float32
    image_array = np.expand_dims(image_array, axis=0)    # Add batch dimension
    return image_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # single output sigmoid

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-8)
    return heatmap.numpy()

# Streamlit UI
st.title("Pneumonia Detection")
st.write("Upload a chest X-ray image to classify it as **Pneumonia** or **Normal**.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    img = Image.open(uploaded_image).convert('RGB')     # Ensure 3 channels
    img = img.resize((224, 224))
    st.image(img, caption="Uploaded Image")

    image_array = preprocess_image(img)
    predictions = model.predict(image_array)
    confidence = float(predictions[0][0])
    predicted_label = "Pneumonia" if confidence > 0.5 else "Normal"
    st.markdown(f"### Prediction: **{predicted_label}**")
    st.write(f"Confidence: {'High' if confidence > 0.75 or confidence < 0.25 else 'Low'} `{confidence:.4f}`")

    # Convert to NumPy array
    img_np = np.array(img)
    # Grad-CAM: image must be 3D (H, W, 3)
    if len(img_np.shape) == 2:  # grayscale safety
        img_np = np.stack([img_np] * 3, axis=-1)
    # Normalize and expand dims for model
    img_array = np.expand_dims(img_np, axis=0) / 255.0
    # Get last conv layer of base model (DenseNet121)
    base_model = model.get_layer('densenet121')
    heatmap = make_gradcam_heatmap(img_array, base_model, last_conv_layer_name='conv5_block16_concat')
    # Overlay heatmap
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    # Display result
    st.image(superimposed_img, caption="Grad-CAM Heatmap", channels="RGB")