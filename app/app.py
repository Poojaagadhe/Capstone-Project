import streamlit as st
import numpy as np
import nibabel as nib
import json
import matplotlib.pyplot as plt
from PIL import Image

from streamlit_image_comparison import image_comparison

from src.inference import load_model, generate_ct
from src.preprocess import preprocess_image


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="MRI → Synthetic CT Generator",
    page_icon="🧠",
    layout="wide"
)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def get_model():
    return load_model()

model = get_model()


# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>MRI → Synthetic CT Generator</h1>
<p style='text-align:center; font-size:18px;'>
Deep Learning System for MRI-only Radiotherapy Planning
</p>
""", unsafe_allow_html=True)

st.divider()


# ---------------- MODEL INFO PANEL ----------------
st.subheader("Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
**Architecture**

Pix2Pix U-Net Generator  
Encoder–Decoder with Skip Connections
""")

with col2:
    st.info("""
**Dataset**

181 paired MRI-CT scans  
21,183 processed slices  
Resolution: 256 × 256
""")

with col3:
    st.info("""
**Training Setup**

Loss: L1 + Feature Matching  
PatchGAN Discriminator  
Framework: PyTorch
""")


# ---------------- PERFORMANCE METRICS ----------------
st.subheader("Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric("SSIM", "0.94")
m2.metric("PSNR", "26.04 dB")
m3.metric("MAE", "39.07 HU")

st.divider()


# ---------------- EXAMPLE RESULTS ----------------
st.subheader("Example Results")

c1, c2 = st.columns(2)

with c1:
    st.image(
        "assets/example_mri.png",
        caption="Input MRI",
        use_container_width=True
    )

with c2:
    st.image(
        "assets/example_ct.png",
        caption="Generated Synthetic CT",
        use_container_width=True
    )

st.divider()


# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Input Type",
    ["Single MRI Image", "MRI Volume (.nii)"]
)

show_heatmap = st.sidebar.checkbox("Show Error Heatmap")
show_dashboard = st.sidebar.checkbox("Show Training Dashboard")


# =====================================================
# SINGLE IMAGE MODE
# =====================================================
if mode == "Single MRI Image":

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["png", "jpg", "jpeg"]
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

        # Comparison slider
        st.subheader("MRI ↔ Synthetic CT Comparison")

        image_comparison(
            img1=image_np,
            img2=ct_img,
            label1="MRI",
            label2="Synthetic CT"
        )

        # Error heatmap
        if show_heatmap:

            st.subheader("Error Heatmap")

            diff = np.abs(ct_img - image_np)

            fig, ax = plt.subplots()
            heat = ax.imshow(diff, cmap="hot")
            fig.colorbar(heat)

            st.pyplot(fig)


# =====================================================
# MRI VOLUME MODE (.nii)
# =====================================================
else:

    uploaded_file = st.file_uploader(
        "Upload MRI Volume",
        type=["nii", "nii.gz"]
    )

    if uploaded_file:

        volume = nib.load(uploaded_file).get_fdata()

        slice_index = st.slider(
            "Select Slice",
            0,
            volume.shape[2] - 1,
            volume.shape[2] // 2
        )

        slice_img = volume[:, :, slice_index]

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

        st.subheader("MRI ↔ Synthetic CT Comparison")

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


# =====================================================
# TRAINING DASHBOARD
# =====================================================
if show_dashboard:

    st.divider()
    st.header("Training Metrics Dashboard")

    try:

        with open("results/training_history.json") as f:
            history = json.load(f)

        with open("results/test_set_results.json") as f:
            results = json.load(f)

        st.subheader("Generator Training Loss")
        st.line_chart(history["generator_loss"])

        st.subheader("Evaluation Metrics")

        metric_cols = st.columns(3)

        metric_cols[0].metric("SSIM", results["SSIM"])
        metric_cols[1].metric("PSNR", results["PSNR"])
        metric_cols[2].metric("MAE", results["MAE"])

    except:
        st.warning("Metrics files not found in results folder.")