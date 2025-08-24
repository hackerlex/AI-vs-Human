import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load model
MODEL_PATH = os.path.join(os.getcwd(), "ResNet152V2-AIvsHumanGenImages.keras")
model = load_model(MODEL_PATH)

st.title("AI vs Human Generated Image Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((512, 512))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    label = "AI Generated" if pred[0][0] > 0.5 else "Human Created"
    st.write("Prediction:", label)
