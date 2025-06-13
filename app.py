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
st.markdown("""
    <style>
        /* Global Background and Font */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #111 !important;
            color: #fff !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1c1c1c !important;
            color: white !important;
        }

        /* Heading & Text */
        h1, h2, h3, h4, h5, h6, p, span, div {
            color: #ffffff !important;
        }

        /* File uploader area */
        .css-1cpxqw2, .css-13sdm1e, .css-1v0mbdj {
            background-color: #222 !important;
            color: white !important;
            border: 1px solid #444;
        }

        /* Warning/success boxes */
        .stAlert {
            border-radius: 8px;
            padding: 10px;
            font-size: 1rem;
        }

        /* Radio buttons */
        .stRadio > label {
            color: white !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #e84d4d;
            color: white !important;
            padding: 0.6rem 1.4rem;
            font-size: 1rem;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(232, 77, 77, 0.4);
            transition: all 0.2s ease-in-out;
        }

        .stButton>button:hover {
            background-color: #ff5e5e;
            transform: scale(1.03);
        }

        /* Image shadow and border */
        img {
            border-radius: 10px;
            box-shadow: 0 0 12px rgba(255, 255, 255, 0.1);
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #ff4b4b, #e84d4d); border-radius: 12px;'>
        <h1 style='color: white; font-size: 2.8em; text-shadow: 2px 2px #000;'>🔍 Transistor Anomaly Detector</h1>
        <p style='color: white; font-size: 1.3em;'>AI-Powered Quality Control for Transistor Circuits</p>
    </div>
""", unsafe_allow_html=True)
with st.sidebar:
    img = Image.open(".docs/overview_dataset.jpg")
    st.image(img.resize((150, 150)))
    st.header("📝 About ")
    st.markdown("""
    This AI-powered app inspects **Transistor Circuit Images** and classifies them as:
    - ✅ Good
    - ⚠️ Anomaly

    **Ideal For:**
    - Semiconductor QC
    - Electronics Inspection
    - Lab Demonstrations
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
