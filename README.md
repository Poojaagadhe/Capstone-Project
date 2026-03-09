# MRI-to-CT Image Translation Using Deep Learning

- Student: POOJA VILAS GADHE
- Student Code: IITMCS_24061184
- Topic: MRI-to-CT Image Translation Using Deep Learning

# Project Overview:
-This capstone project focuses on developing and evaluating deep learning models for automated medical image translation from T1-weighted brain MRI scans to synthetic Computed Tomography (sCT) images.
-The primary goal is to support MRI-only radiotherapy treatment planning. By generating highly accurate sCTs, we can:
-Streamline Workflows: Eliminate the need for dual-imaging (MRI + CT).
-Reduce Patient Burden: Decrease patient exposure to ionizing radiation.
-Enhance Precision: Avoid registration errors inherent in manual image fusion.
-Three distinct deep learning models were implemented and rigorously tested. All three achieved clinically acceptable accuracy (Mean Absolute Error, {MAE} < 100{HU}), demonstrating the strong viability of an MRI-only workflow.

 # Architectures and Key Features:
 --The project evaluated three specialized U-Net variants on a dataset of 181 brain scans.

# 1. U-Net with Local Decoder (Best Performer)
 * Architecture: Dual-component design featuring a standard U-Net encoder coupled with a specialized local decoder.

* Key Insight: Achieved the best overall performance by incorporating refinement layers for enhanced fine anatomical detail preservation, successfully capturing subtle tissue boundaries.

* Loss Function: L1 loss(lambda=1.0) + SSIM loss(lambda=0.5)

* Parameters: approx. 31 million

# 2. U-Net with Pix2Pix PatchGAN Discriminator (Most Consistent)

* Architecture: A sophisticated Adversarial Training approach, combining a U-Net generator with a multi-scale PatchGAN discriminator.

* Key Insight: Demonstrated high consistency and reliability, successfully balancing pixel-wise accuracy with perceptual quality typical of GANs.

* Loss Function: L1 loss(lambda=300) + Feature matching(lambda=20) + Gradient difference loss(lambda=50)

* Parameters: approx 31 million

# 3. Turbo U-Net with Multi-Scale Discriminator (Most Efficient)

* Architecture: An optimized U-Net design incorporating residual blocks for improved gradient flow and an efficient four-stage encoder-decoder structure.

* Key Insight: This architecture is highly computationally efficient with a significantly reduced parameter count, making it suitable for resource-constrained environments.

* Loss Function: L1 loss(lambda=0.5) + SSIM loss(lambda=0.3) + MS-SSIM loss(lambda=0.2) + Adversarial loss(lambda=1.0)

* Parameters: approx 8 million


# Performance Summary
- The models were evaluated on the Test Set (3,120 slices) using standard medical image quality metrics: SSIM, PSNR (dB), and MAE (HU).
1) U-Net + Local Decoder : Test SSIM = 0.8725 (Highest) | Test PSNR (dB) = 25.17 | Test MAE (HU) = 69 
2) U-Net + PatchGAN  :  Test SSIM = 0.93-0.95(range) | Test PSNR (dB) = 26.04 | Test MAE (HU) = 39.07 (lowest)
3) Turbo U-Net   :  Test SSIM = 0.8118 | Test PSNR (dB) = 22.88 | Test MAE (HU) = 95.13

# Performance Insights:
- Best Overall: The U-Net with Local Decoder achieved the highest structural similarity ({SSIM} = 0.8725) and superior detail preservation, but the U-Net with PatchGAN is equally critical, achieving the lowest $\{MAE} = (39.07 HU ) required for precise dose calculation.
- Reliability: The U-Net with PatchGAN demonstrated the most reliable consistency and lowest MAE.
- Efficiency: The Turbo U-Net offers an excellent accuracy-efficiency trade-off, with only 8 million parameters.
- Clinical Viability: All models maintained {MAE} < 100 HU , validating their potential for clinical dose calculation accuracy.

# Dataset and Preprocessing
- Source Data: 181 unique T1-weighted brain MRI and corresponding CT scans in NIFTI (.nii) format.
- Processed Data: 21,183 high-quality 2D slices.
- Critical Splitting: Data splitting was performed at the patient level (70% Train, 15% Validation, 15% Test) to strictly prevent data leakage and ensure that performance metrics reflect true model generalization.
- Preprocessing: Includes image resizing to (256x256) intensity clipping for outliers, and Min-Max normalization to the range {-1, 1}.

# Future Work
- To translate these research findings into clinical practice, the following directions are proposed:
- Clinical Validation Studies: Conduct rigorous trials to assess the performance of sCTs in real-world radiotherapy treatment workflows.
- Dosimetric Validation: Integrate the best-performing models directly with treatment planning systems to confirm that the low {MAE} values translate into acceptable dose distributions.
- Generalizability: Validate the models on large, diverse datasets from multiple institutions to ensure robustness across different scanner types and imaging protocols.
- Anatomical Extension: Adapt and validate the models for other anatomical regions (e.g., pelvis or abdomen) where {MRI-only} workflows are also highly beneficial.
