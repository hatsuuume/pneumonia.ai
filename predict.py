import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Load model
model = load_model('finalmodel.keras')
def get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv5_block3_out"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam /= np.max(cam) if np.max(cam) != 0 else 1
    cam = np.uint8(255 * cam)
    cam = tf.image.resize(np.expand_dims(cam, axis=-1), (224, 224)).numpy()
    cam = np.squeeze(cam)
    return cam

# Class labels must match your training order
class_names = ['Normal', 'Pneumonia']

st.title("Confusion Matrix Generator")

# Let user select test directory
test_dir = st.text_input("Enter path to test image directory (e.g. test/):")

if test_dir and os.path.isdir(test_dir):
    y_true = []
    y_pred = []

    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = np.array(img).astype(np.float32)
                img_array_expanded = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array_expanded)[0][0]
                predicted_class = 1 if prediction > 0.5 else 0

                y_true.append(class_index)
                y_pred.append(predicted_class)

                # Grad-CAM heatmap
                heatmap = get_gradcam_heatmap(model, img_array_expanded)

                # Overlay heatmap on image
                heatmap_colored = plt.cm.jet(heatmap / 255.0)[:, :, :3]  # Remove alpha
                overlay = 0.5 * heatmap_colored + img_array / 255.0
                overlay = np.clip(overlay, 0, 1)

                # Display side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original", use_column_width=True)
                with col2:
                    st.image(overlay, caption="Prediction Heatmap", use_column_width=True)

            except Exception as e:
                st.warning(f"Failed to load {img_path}: {e}")

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    st.pyplot(fig)

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    st.markdown(f"**Accuracy: {acc * 100:.2f}%**")

else:
    st.info("Please enter a valid directory with class subfolders.")
