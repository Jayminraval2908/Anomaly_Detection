import streamlit as st
import torch
from PIL import Image
import numpy as np
import os

from tensorflow.keras.models import load_model
import tensorflow as tf


img_to_array = tf.keras.utils.img_to_array

import torchvision.transforms as transforms

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Page layout
st.set_page_config(page_title="Transistor Anomaly Detector", page_icon="🔍")
# Banner
st.markdown("""
    <div style='text-align: center; padding: 2rem; background-color: #2c2b4b; border-radius: 10px;'>
        <h1 style='color: #f8f8f8; font-size: 2.5em;'>🔍 Transistor Anomaly Detector</h1>
        <p style='color: #e0e0e0; font-size: 1.2em;'>AI-Powered Quality Control for Transistor circuits</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar background
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #2c2b4b !important;
            color: white;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li {
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)
with st.sidebar:
    img = Image.open(".docs/overview_dataset.jpg")
    st.image(img.resize((150, 150)))
    st.header("📝 About ")
    st.markdown("""
    Transistor Anomaly Detection App is a powerful AI-powered application designed to help businesses and engineers streamline quality control for **Transistor Circuit Inspections**.

    This app uses deep learning and computer vision to **Automatically classify Transistor circuit images** as:
    - ✅ Good
    - ⚠️ Anomaly (defect or irregularity)

    Ideal for:
    - Semiconductor manufacturing
    - Electronics quality assurance
    - Educational labs
    """)
# Image loading
# Image loading
def load_uploaded_image(file):
    img = Image.open(file).convert("RGB")
    return img

st.subheader("Select Image Input Method")
input_method = st.radio("options", ["📁 File Uploader", "📷 Camera Input"], label_visibility="collapsed")

uploaded_file_img = None
camera_file_img = None

if input_method == "📁 File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Please upload a Transistor circuit image.")

elif input_method == "📷 Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Capture a Transistor Circuit")
    if camera_image_file is not None:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Captured Image", width=300)
        st.success("Image captured successfully!")
    else:
        st.warning("Please capture an image.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Keras model safely using tensorflow.keras
try:
    model = load_model("keras_model.h5", compile=False)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# You can modify the labels file if needed
class_names = ['Good', 'Anomaly']

# Anomaly Detection Logic
def Anomaly_Detection(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]

    if predicted_class == "Good":
        return "✅ Congratulations! Transistor is classified as Good with no anomalies."
    else:
        return "⚠️ Alert! An Anomaly has been detected in the Transistor."

# Prediction Trigger
submit = st.button(label="Submit for Inspection")

if submit:
    st.subheader("🧠 AI Inspection Result")
    img_file = uploaded_file_img if input_method == "📁 File Uploader" else camera_file_img

    if img_file is not None:
        with st.spinner("Inspecting the Transistor image..."):
            prediction = Anomaly_Detection(img_file)
            if "Anomaly" in prediction:
                st.error(prediction)
            else:
                st.success(prediction)
    else:
        st.warning("Please provide an image before submitting.")
# 🧾 Footer
st.markdown("""<hr style="margin-top: 3rem;">""", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; font-size: 0.85em; color: grey;'>
        👨‍💻 Developed by <a href='https://github.com/Jayminraval2908/Anomaly_Detection' target='_blank'>Jaymin Raval</a>
    </div>
""", unsafe_allow_html=True)
