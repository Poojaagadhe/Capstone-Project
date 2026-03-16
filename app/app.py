import streamlit as st
import numpy as np
from PIL import Image

from src.inference import load_model, generate_ct
from src.preprocess import preprocess_image

st.title("MRI → Synthetic CT Generator")

model = load_model()

uploaded_file = st.file_uploader("Upload MRI image", type=["png","jpg","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    input_tensor = preprocess_image(image_np)

    output = generate_ct(model, input_tensor)

    output_image = output.squeeze().numpy()

    st.subheader("Input MRI")
    st.image(image_np, clamp=True)

    st.subheader("Generated CT")
    st.image(output_image, clamp=True)
