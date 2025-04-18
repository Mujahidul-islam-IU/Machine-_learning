import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.express as px

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="🧠 Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
)

# ------------------- CUSTOM CSS -------------------
def local_css():
    st.markdown("""
        <style>
            body {
                background-color: #f4f4f4;
            }
            .main {
                background-color: #ffffff;
                padding: 2rem;
                border-radius: 15px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                height: 3em;
                width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)

local_css()

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_brain_tumor_model():
    return load_model("model/brain_tumor_model_v2.keras")

model = load_brain_tumor_model()

# ------------------- CONSTANTS -------------------
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# ------------------- IMAGE PREPROCESS -------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((128, 128))  # Ensure matches model
    image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# ------------------- APP TITLE -------------------
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🧠 Brain Tumor Classifier")
st.write("Upload a brain MRI scan to detect tumor type using a deep learning model.")

# ------------------- IMAGE UPLOAD -------------------
uploaded_file = st.file_uploader("📤 Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)
    with col2:
        st.markdown("**Image Info**")
        st.write(f"📁 File name: `{uploaded_file.name}`")
        st.write(f"🖼 Size: `{image.size}`")
        st.write(f"🎨 Mode: `{image.mode}`")

    st.markdown("---")
    with st.spinner("🔍 Analyzing MRI..."):
        preprocessed = preprocess_image(image)
        prediction_probs = model.predict(preprocessed)[0]
        predicted_class = class_names[np.argmax(prediction_probs)]
        confidence = np.max(prediction_probs)

    # ------------------- RESULT -------------------
    st.success(f"🎯 **Prediction: {predicted_class.upper()}**")
    st.info(f"Confidence: `{confidence:.2f}`")

    # ------------------- PLOT -------------------
    fig = px.bar(
        x=class_names,
        y=prediction_probs,
        labels={'x': "Tumor Type", 'y': "Probability"},
        color=class_names,
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Prediction Probabilities"
    )
    fig.update_layout(xaxis_title=None, yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)
