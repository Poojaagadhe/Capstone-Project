import streamlit as st
import numpy as np
import nibabel as nib
import json
import os
import random
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
st.markdown(
"""
<h1 style='text-align:center;'>MRI → Synthetic CT Generator</h1>
<p style='text-align:center; font-size:18px;'>
Deep Learning System for MRI-only Radiotherapy Planning
</p>
""",
unsafe_allow_html=True
)

st.divider()


# ---------------- TABS ----------------
tab_demo, tab_examples, tab_dashboard, tab_info = st.tabs(
    ["Demo", "Examples", "Training Dashboard", "Model Info"]
)


# ======================================================
# DEMO TAB
# ======================================================
with tab_demo:

    st.subheader("Upload MRI")

    uploaded_file = st.file_uploader(
        "Upload MRI Image (.png) or MRI Volume (.nii)",
        type=["png", "nii", "nii.gz"]
    )

    show_heatmap = st.checkbox("Show Error Heatmap")

    if uploaded_file:

        file_name = uploaded_file.name.lower()

        # ---------- PNG IMAGE ----------
        if file_name.endswith(".png"):

            image = Image.open(uploaded_file).convert("L")
            image_np = np.array(image)

            tensor = preprocess_image(image_np)

            with st.spinner("Generating synthetic CT..."):
                output = generate_ct(model, tensor)

            ct_img = output.squeeze().numpy()

            col1, col2 = st.columns(2)

            with col1:
                st.image(image_np, caption="Input MRI", use_container_width=True)

            with col2:
                st.image(ct_img, caption="Generated CT", use_container_width=True)

            st.subheader("MRI ↔ Synthetic CT Comparison")

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


        # ---------- NII VOLUME ----------
        else:

            volume = nib.load(uploaded_file).get_fdata()

            slice_index = st.slider(
                "Select MRI Slice",
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
                st.image(slice_img, caption="MRI Slice", use_container_width=True)

            with col2:
                st.image(ct_img, caption="Generated CT", use_container_width=True)

            st.subheader("MRI ↔ Synthetic CT Comparison")

            image_comparison(
                img1=slice_img,
                img2=ct_img,
                label1="MRI",
                label2="Synthetic CT"
            )


# ======================================================
# EXAMPLES TAB
# ======================================================
with tab_examples:

    st.subheader("Example Results")

    results_dir = "results/pix2pix_unet"

    try:

        samples = [f for f in os.listdir(results_dir) if f.endswith(".png")]

        if "example_sample" not in st.session_state:
            st.session_state.example_sample = random.choice(samples)

        sample_file = st.session_state.example_sample

        example_img = Image.open(os.path.join(results_dir, sample_file))
        example_np = np.array(example_img)

        width = example_np.shape[1]
        third = width // 3

        mri = example_np[:, :third]
        gt_ct = example_np[:, third:2*third]
        pred_ct = example_np[:, 2*third:]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(mri, caption="Input MRI", use_container_width=True)

        with col2:
            st.image(gt_ct, caption="Ground Truth CT", use_container_width=True)

        with col3:
            st.image(pred_ct, caption="Predicted CT", use_container_width=True)

        if st.button("Show Another Example", key="example_button"):
           st.session_state.example_sample = random.choice(samples)
           st.rerun()

    except:
        st.warning("Example test samples not found.")


# ======================================================
# TRAINING DASHBOARD TAB
# ======================================================
with tab_dashboard:

    st.subheader("Training Metrics")

    history_path = "results/training_history.json"
    results_path = "results/test_set_results.json"

    if os.path.exists(history_path) and os.path.exists(results_path):

        with open(history_path) as f:
            history = json.load(f)

        with open(results_path) as f:
            results = json.load(f)

        # ---------------- Training Loss ----------------
        if "generator_loss" in history:
            st.line_chart(history["generator_loss"])

        # ---------------- Metrics ----------------
        ssim_mean = results.get("ssim", {}).get("mean")
        ssim_std  = results.get("ssim", {}).get("std")

        psnr_mean = results.get("psnr", {}).get("mean")
        psnr_std  = results.get("psnr", {}).get("std")

        mae_mean = results.get("mae", {}).get("mean")
        mae_std  = results.get("mae", {}).get("std")

        m1, m2, m3 = st.columns(3)

        if ssim_mean is not None:
            m1.metric("SSIM", f"{ssim_mean:.3f}", f"±{ssim_std:.3f}")
        else:
            m1.metric("SSIM", "N/A")

        if psnr_mean is not None:
            m2.metric("PSNR", f"{psnr_mean:.2f} dB", f"±{psnr_std:.2f}")
        else:
            m2.metric("PSNR", "N/A")

        if mae_mean is not None:
            m3.metric("MAE", f"{mae_mean:.2f} HU", f"±{mae_std:.2f}")
        else:
            m3.metric("MAE", "N/A")

    else:
        st.info("Training metrics file not found in results folder.")

# ======================================================
# MODEL INFO TAB
# ======================================================
with tab_info:

    st.subheader("Model Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
"""
**Architecture**

Pix2Pix U-Net Generator  
Encoder–Decoder with Skip Connections
"""
        )

    with c2:
        st.info(
"""
**Dataset**

181 MRI–CT paired scans  
21,183 slices  
Resolution: 256 × 256
"""
        )

    with c3:
        st.info(
"""
**Training**

Loss: L1 + Feature Matching  
PatchGAN Discriminator  
Framework: PyTorch
"""
        )







