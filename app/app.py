import streamlit as st
import numpy as np
import nibabel as nib
import json
from PIL import Image
import matplotlib.pyplot as plt

from streamlit_image_comparison import image_comparison

from src.inference import load_model, generate_ct
from src.preprocess import preprocess_image


st.set_page_config(
    page_title="MRI → Synthetic CT Generator",
    layout="wide",
    page_icon="🧠"
)


# -------- Model Loading --------
@st.cache_resource
def get_model():
    return load_model()

model = get_model()


st.title("MRI → Synthetic CT Generator")
st.caption("Deep Learning Based MRI-only Radiotherapy Planning")

st.divider()


# -------- Sidebar --------
st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Input Type",
    ["Single Image", "MRI Volume (.nii)"]
)

show_heatmap = st.sidebar.checkbox("Show Error Heatmap")
show_dashboard = st.sidebar.checkbox("Show Training Dashboard")


# ====================================
# MRI VOLUME VIEWER
# ====================================

if mode == "MRI Volume (.nii)":

    uploaded_file = st.file_uploader("Upload MRI Volume", type=["nii", "nii.gz"])

    if uploaded_file:

        volume = nib.load(uploaded_file).get_fdata()

        slice_index = st.slider(
            "Select Slice",
            0,
            volume.shape[2]-1,
            volume.shape[2]//2
        )

        slice_img = volume[:,:,slice_index]

        tensor = preprocess_image(slice_img)

        with st.spinner("Generating synthetic CT..."):
            output = generate_ct(model, tensor)

        ct_img = output.squeeze().numpy()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("MRI Slice")
            st.image(slice_img, clamp=True)

        with col2:
            st.subheader("Generated CT")
            st.image(ct_img, clamp=True)

        image_comparison(
            img1=slice_img,
            img2=ct_img,
            label1="MRI",
            label2="Synthetic CT"
        )

        if show_heatmap:

            st.subheader("Error Heatmap")

            diff = np.abs(ct_img - slice_img)

            fig, ax = plt.subplots()

            heat = ax.imshow(diff, cmap="hot")
            fig.colorbar(heat)

            st.pyplot(fig)


# ====================================
# SINGLE IMAGE MODE
# ====================================

else:

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["png","jpg","jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("L")
        image_np = np.array(image)

        tensor = preprocess_image(image_np)

        with st.spinner("Generating synthetic CT..."):
            output = generate_ct(model, tensor)

        ct_img = output.squeeze().numpy()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input MRI")
            st.image(image_np, clamp=True)

        with col2:
            st.subheader("Generated CT")
            st.image(ct_img, clamp=True)

        image_comparison(
            img1=image_np,
            img2=ct_img,
            label1="MRI",
            label2="Synthetic CT"
        )

        if show_heatmap:

            st.subheader("Error Heatmap")

            diff = np.abs(ct_img - image_np)

            fig, ax = plt.subplots()

            heat = ax.imshow(diff, cmap="hot")
            fig.colorbar(heat)

            st.pyplot(fig)


# ====================================
# TRAINING DASHBOARD
# ====================================

if show_dashboard:

    st.divider()
    st.header("Training Metrics Dashboard")

    try:

        with open("results/training_history.json") as f:
            history = json.load(f)

        with open("results/test_set_results.json") as f:
            results = json.load(f)

        st.subheader("Training Loss")

        st.line_chart(history["generator_loss"])

        st.subheader("Validation Metrics")

        metrics = {
            "SSIM": results["SSIM"],
            "PSNR": results["PSNR"],
            "MAE": results["MAE"]
        }

        st.json(metrics)

    except:
        st.warning("Metrics files not found in results folder.")