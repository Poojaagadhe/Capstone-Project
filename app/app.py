import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import numpy as np
import nibabel as nib
import json
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
import time

from streamlit_image_comparison import image_comparison

from src.inference import load_model, generate_ct
from src.preprocess import preprocess_image


# -------------------------------------------------------
# SAFE NORMALIZATION FUNCTION
# -------------------------------------------------------
def normalize_image(img):

    img = img.astype("float32")

    min_val = np.min(img)
    max_val = np.max(img)

    if max_val - min_val < 1e-8:
        return np.zeros_like(img)

    return (img - min_val) / (max_val - min_val)


# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="MRI → Synthetic CT Generator",
    page_icon="🧠",
    layout="wide"
)

torch.set_num_threads(1)


# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def get_model():
    return load_model()

model = get_model()


# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.title("MRI → Synthetic CT Generator")
st.caption("Deep learning system for MRI-only radiotherapy planning")

st.divider()


# -------------------------------------------------------
# TABS
# -------------------------------------------------------
tab_demo, tab_examples, tab_dashboard, tab_info = st.tabs(
    ["Demo", "Examples", "Training Dashboard", "Model Info"]
)


# =======================================================
# DEMO TAB
# =======================================================
with tab_demo:

    st.subheader("Upload MRI")

    uploaded_file = st.file_uploader(
        "Upload MRI Image (.png) or MRI Volume (.nii)",
        type=["png", "nii", "nii.gz"]
    )

    show_heatmap = st.checkbox("Show Error Heatmap")

    if uploaded_file:

        filename = uploaded_file.name.lower()

        # ------------------------------------------------
        # PNG MRI
        # ------------------------------------------------
        if filename.endswith(".png"):

            image = Image.open(uploaded_file).convert("L")
            image_np = np.array(image)

            # resize for model
            image_np = np.array(
                Image.fromarray(image_np).resize((256,256))
            )

            image_np = normalize_image(image_np)

            tensor = preprocess_image(image_np)

            start = time.time()

            with st.spinner("Generating synthetic CT..."):
                output = generate_ct(model, tensor)

            inference_time = time.time() - start

            ct_img = output.squeeze().detach().cpu().numpy()
            ct_img = normalize_image(ct_img)

            # ensure same size
            if image_np.shape != ct_img.shape:
                image_np = np.array(
                    Image.fromarray(image_np).resize(ct_img.shape[::-1])
                )

            # convert for display
            mri_display = (image_np * 255).astype(np.uint8)
            ct_display = (ct_img * 255).astype(np.uint8)

            col1, col2 = st.columns(2)

            with col1:
                st.image(mri_display, caption="Input MRI", width="stretch")

            with col2:
                st.image(ct_display, caption="Generated CT", width="stretch")

            st.success(f"Inference time: {inference_time:.3f} seconds")

            image_comparison(
                img1=mri_display,
                img2=ct_display,
                label1="MRI",
                label2="Synthetic CT"
            )

            # download CT
            st.download_button(
                "Download Generated CT",
                data=Image.fromarray(ct_display).tobytes(),
                file_name="synthetic_ct.png"
            )

            # heatmap
            if show_heatmap:

                diff = np.abs(ct_img - image_np)

                fig, ax = plt.subplots()
                heat = ax.imshow(diff, cmap="hot")
                fig.colorbar(heat)

                st.pyplot(fig)
                plt.close()


        # ------------------------------------------------
        # NIfTI MRI
        # ------------------------------------------------
        else:

            volume = nib.load(uploaded_file).get_fdata()

            slice_index = st.slider(
                "Select MRI Slice",
                0,
                volume.shape[2] - 1,
                volume.shape[2] // 2
            )

            slice_img = volume[:, :, slice_index]

            slice_img = np.array(
                Image.fromarray(slice_img).resize((256,256))
            )

            slice_img = normalize_image(slice_img)

            tensor = preprocess_image(slice_img)

            start = time.time()

            with st.spinner("Generating synthetic CT..."):
                output = generate_ct(model, tensor)

            inference_time = time.time() - start

            ct_img = output.squeeze().detach().cpu().numpy()
            ct_img = normalize_image(ct_img)

            if slice_img.shape != ct_img.shape:
                slice_img = np.array(
                    Image.fromarray(slice_img).resize(ct_img.shape[::-1])
                )

            mri_display = (slice_img * 255).astype(np.uint8)
            ct_display = (ct_img * 255).astype(np.uint8)

            col1, col2 = st.columns(2)

            with col1:
                st.image(mri_display, caption="MRI Slice", width="stretch")

            with col2:
                st.image(ct_display, caption="Generated CT", width="stretch")

            st.success(f"Inference time: {inference_time:.3f} seconds")

            image_comparison(
                img1=mri_display,
                img2=ct_display,
                label1="MRI",
                label2="Synthetic CT"
            )


# =======================================================
# EXAMPLES TAB
# =======================================================
with tab_examples:

    st.subheader("Example Results")

    results_dir = "results/pix2pix_unet"

    if os.path.exists(results_dir):

        samples = [f for f in os.listdir(results_dir) if f.endswith(".png")]

        if samples:

            if "example_sample" not in st.session_state:
                st.session_state.example_sample = random.choice(samples)

            sample_file = st.session_state.example_sample
            img_path = os.path.join(results_dir, sample_file)

            example_img = Image.open(img_path).convert("RGB")
            example_np = np.array(example_img)

            width = example_np.shape[1]
            third = width // 3

            mri = example_np[:, :third]
            gt_ct = example_np[:, third:2*third]
            pred_ct = example_np[:, 2*third:]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(mri, caption="Input MRI", width="stretch")

            with col2:
                st.image(gt_ct, caption="Ground Truth CT", width="stretch")

            with col3:
                st.image(pred_ct, caption="Predicted CT", width="stretch")

            if st.button("Show Another Example"):
                st.session_state.example_sample = random.choice(samples)
                st.rerun()


# =======================================================
# TRAINING DASHBOARD
# =======================================================
with tab_dashboard:

    st.subheader("Training Metrics")

    history_path = "results/training_history.json"
    results_path = "results/test_set_results.json"

    if os.path.exists(history_path) and os.path.exists(results_path):

        with open(history_path) as f:
            history = json.load(f)

        with open(results_path) as f:
            results = json.load(f)

        if "generator_loss" in history:
            st.line_chart(history["generator_loss"])

        ssim_mean = results.get("ssim", {}).get("mean")
        psnr_mean = results.get("psnr", {}).get("mean")
        mae_mean = results.get("mae", {}).get("mean")

        c1, c2, c3 = st.columns(3)

        if ssim_mean:
            c1.metric("SSIM", f"{ssim_mean:.3f}")

        if psnr_mean:
            c2.metric("PSNR", f"{psnr_mean:.2f} dB")

        if mae_mean:
            c3.metric("MAE", f"{mae_mean:.2f} HU")


# =======================================================
# MODEL INFO
# =======================================================
with tab_info:

    st.subheader("Model Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info(
"""
Architecture  
Pix2Pix U-Net Generator  
Encoder-decoder with skip connections
"""
        )

    with c2:
        st.info(
"""
Dataset  
181 MRI-CT scans  
21,183 slices  
Resolution: 256×256
"""
        )

    with c3:
        st.info(
"""
Training  
L1 + Feature Matching loss  
PatchGAN discriminator  
Framework: PyTorch
"""
        )