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
    <div style='text-align: center; padding: 2rem; background-color: #f0f4ff; border-radius: 10px;'>
        <h1 style='color: #333; font-size: 2.5em;'>🔍 Transistor Anomaly Detector</h1>
        <p style='color: #333; font-size: 1.2em;'>AI-Powered Quality Control for Transistor circuits</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar background
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #f0f4ff !important;
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
st.subheader("Select Image Input Method")
input_method = st.radio("options", ["📁 File Uploader", "📷 Camera Input"], label_visibility="collapsed")

uploaded_file_img = None
camera_file_img = None

def load_uploaded_image(file):
    return Image.open(file).convert("RGB")

if input_method == "📁 File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        uploaded_file_img = load_uploaded_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")
    else:
        st.warning("Upload a transistor circuit image.")
elif input_method == "📷 Camera Input":
    camera_image_file = st.camera_input("Capture a Transistor Circuit")
    if camera_image_file:
        camera_file_img = load_uploaded_image(camera_image_file)
        st.image(camera_file_img, caption="Captured Image", width=300)
        st.success("Image captured successfully!")
    else:
        st.warning("Please capture an image.")

# 🧠 Load Keras model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = load_model("keras_model.h5", compile=False)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

class_names = ['Good', 'Anomaly']

# 🔍 Anomaly detection logic
def Anomaly_Detection(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]

    return ("✅ Transistor is Good. No anomalies detected."
            if predicted_class == "Good"
            else "⚠️ Alert! An Anomaly detected in the Transistor.")

# 🚀 Submit button
submit = st.button("🚀 Submit for Inspection")

if submit:
    st.subheader("🧠 AI Inspection Result")
    img_file = uploaded_file_img or camera_file_img
    if img_file:
        with st.spinner("🔍 AI is analyzing the transistor..."):
            prediction = Anomaly_Detection(img_file)
            st.success(prediction) if "Good" in prediction else st.error(prediction)
    else:
        st.warning("Please provide an image before submitting.")

# 🧾 Footer
st.markdown("""<hr style="margin-top: 3rem;">""", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; font-size: 0.85em; color: grey;'>
        👨‍💻 Developed by <a href='https://github.com/Jayminraval2908/Anomaly_Detection' target='_blank'>Jaymin Raval</a>
    </div>
""", unsafe_allow_html=True)
