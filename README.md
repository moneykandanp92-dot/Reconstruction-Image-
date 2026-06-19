# Computer Vision and Diffusion-Based Reconstruction of Structural Damage Images

## Overview
This project presents an AI-powered framework for structural damage assessment, reconstruction, and repair estimation using computer vision and deep learning techniques. The system detects structural damage such as major cracks, minor cracks, and spalling, estimates the damaged area, reconstructs the damaged regions, and provides repair planning information.

The proposed framework combines image classification, semantic segmentation, image reconstruction, and rule-based estimation to support intelligent infrastructure maintenance and disaster recovery applications.

## Problem Statement
Structural damage in buildings caused by aging, environmental conditions, earthquakes, floods, and material degradation can lead to serious safety risks. Traditional inspection methods are manual, time-consuming, and dependent on expert availability.

This project aims to automate damage assessment using deep learning and provide accurate damage reconstruction and repair estimation.
<img width="1463" height="975" alt="image" src="https://github.com/user-attachments/assets/4ab3dd8c-2fb8-42f1-83f8-9c5923b7f1d9" />

## Objectives
* Detect structural damage from images.
* Classify damage into:
  * Major Crack
  * Minor Crack
  * Spalling
* Segment damaged regions and estimate damage area.
* Reconstruct damaged structures using deep learning models.
* Estimate repair requirements such as materials and manpower.
* Support future smart infrastructure maintenance systems.

## System Architecture

### Overall Framework

<img width="1463" height="975" alt="image" src="https://github.com/user-attachments/assets/bc91d906-c8f4-4479-8702-9ebce3ff245c" />



### CNN-Based Damage Classification

<img width="1303" height="525" alt="image" src="https://github.com/user-attachments/assets/294dfebe-3873-4947-bbe2-0e7c2b08211f" />


### U-Net Semantic Segmentation for Damage Area Estimation

<img width="1277" height="501" alt="image" src="https://github.com/user-attachments/assets/aa5f9de5-4f98-4e1f-9165-82d0a37b88c5" />


### Reconstruction Model Comparison
* Autoencoder (AE)
* Denoising Autoencoder (DAE)
* U-Net
* Simplified GAN
* Pix2Pix
<img width="1409" height="517" alt="image" src="https://github.com/user-attachments/assets/b2fa77a6-6fdf-499c-bc31-087dc77adafb" />


---

## Dataset

The dataset consists of structural damage images collected under three categories:
<img width="827" height="390" alt="image" src="https://github.com/user-attachments/assets/13b6b8c6-b467-4fdf-bdb4-02c89349fdcf" />

* Major Crack
* Minor Crack
* Spalling

### Dataset Summary

* Total Images: 248
* Training Images: 184
* Testing Images: 64
* Train-Test Split: 74:26

### Preprocessing

* Image Resizing (128×128)
* Normalization
* Grayscale Conversion
* Gaussian Noise Addition

### Data Augmentation

* Rotation (20°)
* Horizontal Flip
* Brightness Enhancement
* Gaussian Noise Augmentation


## Training Configuration

| Parameter     | Value              |
| ------------- | ------------------ |
| Optimizer     | Adam               |
| Learning Rate | 0.001              |
| Epochs        | 50                 |
| Batch Size    | 8                  |
| Framework     | TensorFlow / Keras |


## Evaluation Metrics

The models were evaluated using:

### Reconstruction Metrics

* Mean Squared Error (MSE)
* Peak Signal-to-Noise Ratio (PSNR)
* Structural Similarity Index (SSIM)
<img width="1484" height="1000" alt="image" src="https://github.com/user-attachments/assets/740582b2-57ae-4217-9ca0-4fabc8eb5e9f" />

### Segmentation Metrics

* Intersection over Union (IoU)
* Dice Coefficient

### Confusion Matrix Metrics
<img width="1463" height="975" alt="image" src="https://github.com/user-attachments/assets/a76cc1c9-3e16-430e-a2ce-5e4bbe2190b4" />

* True Positive (TP)
* True Negative (TN)
* False Positive (FP)
* False Negative (FN)
<img width="1426" height="982" alt="image" src="https://github.com/user-attachments/assets/e3b5baf8-1e9a-425e-90fe-b8def371ba28" />

## Results
<img width="877" height="572" alt="image" src="https://github.com/user-attachments/assets/af2113c2-1e10-4124-bad1-97891a2a55cc" />

### Reconstruction Performance

| Model          | MSE    | PSNR  | SSIM | IoU    | Dice   |
| -------------- | ------ | ----- | ---- | ------ | ------ |
| Autoencoder    | 0.0008 | 23.89 | 0.71 | 0.7643 | 0.8664 |
| Denoising AE   | 0.0011 | 22.71 | 0.60 | 0.7459 | 0.8545 |
| U-Net          | 0.0005 | 31.37 | 0.92 | 0.8811 | 0.9368 |
| Simplified GAN | 0.0009 | 27.13 | 0.79 | 0.8385 | 0.9121 |
| Pix2Pix        | 0.0009 | 24.09 | 0.69 | 0.7959 | 0.8863 |

### Best Model

U-Net achieved the best overall performance with:

* Lowest MSE
* Highest PSNR
* Highest SSIM
* Highest IoU
* Highest Dice Coefficient
<img width="1476" height="964" alt="image" src="https://github.com/user-attachments/assets/3c7d7db9-5041-4309-a288-c8c770f7ce8f" />


## Future Scope

* Real-time mobile application development
* Smart infrastructure monitoring
* Automated repair estimation
* Worker recommendation system
* Integration with location-based service platforms
* AI-powered construction maintenance ecosystem

---

## Startup Vision

This research serves as the foundation for a future AI-powered construction service platform similar to Uber/Ola, where users can:

1. Upload structural damage images.
2. Receive automated damage assessment.
3. Obtain repair material and manpower estimates.
4. Connect with nearby masons, plumbers, and electricians based on location, language preference, and availability.

---

## Applications

* Building Inspection
* Infrastructure Maintenance
* Smart Cities
* Disaster Recovery Planning
* Insurance Assessment
* Construction Site Monitoring

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## Conference Publication

Accepted at:

**1st International Conference on Advances in Thermal and Fluid Systems (ICATFS 2026)**

CGC University, Mohali, Punjab, India

10–11 June 2026


