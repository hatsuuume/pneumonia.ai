import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# Load model
model = load_model('finalmodel.keras')  # replace with your actual model path

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
                #img_array = image.img_to_array(img)
                #img_array = np.expand_dims(img_array, axis=0)
                #
                img_array = np.array(img).astype(np.float32)  # Convert to float32
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                prediction = model.predict(img_array)[0][0]  # sigmoid output
                predicted_class = 1 if prediction > 0.5 else 0

                y_true.append(class_index)
                y_pred.append(predicted_class)
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
