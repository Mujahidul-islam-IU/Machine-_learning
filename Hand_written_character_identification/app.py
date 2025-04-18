import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2
import string

# Load the uploaded Keras model
model = tf.keras.models.load_model("best_model.keras")

# Create labels (0–9 and A–Z)
labels = [str(i) for i in range(10)] + list(string.ascii_uppercase)

st.set_page_config(page_title="Handwritten Character Recognition")
st.title("✍️ Handwritten Character Recognition")
st.write("Upload an image or draw a character below:")

tab1, tab2 = st.tabs(["📤 Upload Image", "🖌️ Draw Character"])

def preprocess_image(img):
    img = ImageOps.grayscale(img)
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        preprocessed = preprocess_image(image)
        prediction = model.predict(preprocessed)
        predicted_class = labels[np.argmax(prediction)]
        st.success(f"### Prediction: {predicted_class}")

with tab2:
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGBA2GRAY)
        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img)
        predicted_class = labels[np.argmax(prediction)]
        st.success(f"### Prediction: {predicted_class}")
