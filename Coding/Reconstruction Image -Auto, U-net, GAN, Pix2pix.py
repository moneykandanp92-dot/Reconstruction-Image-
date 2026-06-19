#!/usr/bin/env python
# coding: utf-8

# In[5]:


# =========================================
# STEP 1 : LOAD DATASET
# =========================================

import os
import cv2
import numpy as np

# =========================================
# IMAGE SIZE
# =========================================

IMG_SIZE = 128

# =========================================
# DATASET PATH
# =========================================

TRAIN_DIR = r"C:\Users\money\OneDrive\Desktop\Reconstruction dataset\train"

# =========================================
# LOAD DATASET
# =========================================

def load_dataset(base_dir):

    images = []

    labels = []

    categories = os.listdir(base_dir)

    for category in categories:

        category_path = os.path.join(
            base_dir,
            category
        )

        print("\nLoading :", category)

        image_count = 0

        for image_name in os.listdir(category_path):

            image_path = os.path.join(
                category_path,
                image_name
            )

            image = cv2.imread(image_path)

            if image is None:
                continue

            # Resize

            image = cv2.resize(
                image,
                (IMG_SIZE, IMG_SIZE)
            )

            # Normalize

            image = image / 255.0

            images.append(image)

            labels.append(category)

            image_count += 1

        print("Images Found :", image_count)

    return np.array(images), labels

# =========================================
# LOAD DATA
# =========================================

processed_images, image_labels = load_dataset(TRAIN_DIR)

# =========================================
# DISPLAY RESULTS
# =========================================

print("\n================================")

print("Dataset Loaded Successfully")

print("================================")

print("Dataset Shape :",
      processed_images.shape)

print("Total Images  :",
      len(processed_images))


# In[7]:


# =========================================
# STEP 2 : LOAD MANUAL TEST IMAGE
# =========================================

import cv2
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# IMAGE PATH
# =========================================

test_image_path = r"C:\Users\money\OneDrive\Desktop\Reconstruction dataset\test\major crack\majorcrack (1).jpeg"

# Example:
# r"C:\Users\money\OneDrive\Desktop\Reconstruction dataset\test\minor crack\image1.jpg"

# =========================================
# LOAD IMAGE
# =========================================

test_image = cv2.imread(test_image_path)

# -----------------------------------------
# CHECK IMAGE
# -----------------------------------------

if test_image is None:

    print("Image Not Found")

    print("Check File Path")

else:

    print("Image Loaded Successfully")

    # =====================================
    # BGR → RGB
    # =====================================

    test_image = cv2.cvtColor(
        test_image,
        cv2.COLOR_BGR2RGB
    )

    # =====================================
    # RESIZE
    # =====================================

    test_image = cv2.resize(
        test_image,
        (128,128)
    )

    # =====================================
    # NORMALIZE
    # =====================================

    test_image = test_image / 255.0

    # =====================================
    # ADD BATCH DIMENSION
    # =====================================

    test_input = np.expand_dims(
        test_image,
        axis=0
    )

    # =====================================
    # DISPLAY IMAGE
    # =====================================

    plt.figure(figsize=(5,5))

    plt.imshow(test_image)

    plt.title("Manual Test Image")

    plt.axis("off")

    plt.show()

    print("Test Image Shape :",
          test_image.shape)

    print("Input Shape :",
          test_input.shape)


# In[8]:


# =========================================
# =========================================
# STEP 2 : DATA AUGMENTATION
# =========================================

import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================================
# SELECT SAMPLE IMAGE
# =========================================

sample_image = processed_images[0]

# =========================================
# AUGMENTATION PARAMETERS
# =========================================

rotation_degree = 20

brightness_factor = 1.3

noise_std = 0.05

# =========================================
# ROTATION
# =========================================

rotation_matrix = cv2.getRotationMatrix2D(
    (64,64),
    rotation_degree,
    1
)

rotated_image = cv2.warpAffine(
    sample_image,
    rotation_matrix,
    (128,128)
)

# =========================================
# HORIZONTAL FLIP
# =========================================

flipped_image = cv2.flip(
    sample_image,
    1
)

# =========================================
# BRIGHTNESS
# =========================================

bright_image = np.clip(
    sample_image * brightness_factor,
    0,
    1
)

# =========================================
# GAUSSIAN NOISE
# =========================================

noise = np.random.normal(
    0,
    noise_std,
    sample_image.shape
)

noisy_image = np.clip(
    sample_image + noise,
    0,
    1
)

# =========================================
# DISPLAY RESULTS
# =========================================

plt.figure(figsize=(15,8))

# Original

plt.subplot(2,3,1)

plt.imshow(sample_image)

plt.title("Original")

plt.axis("off")

# Rotated

plt.subplot(2,3,2)

plt.imshow(rotated_image)

plt.title("Rotated")

plt.axis("off")

# Flipped

plt.subplot(2,3,3)

plt.imshow(flipped_image)

plt.title("Flipped")

plt.axis("off")

# Brightness

plt.subplot(2,3,4)

plt.imshow(bright_image)

plt.title("Brightness")

plt.axis("off")

# Noise

plt.subplot(2,3,5)

plt.imshow(noisy_image)

plt.title("Gaussian Noise")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY PARAMETERS
# =========================================

print("Rotation Degree     :",
      rotation_degree)

print("Brightness Factor   :",
      brightness_factor)

print("Noise Standard Dev  :",
      noise_std)

print("\nData Augmentation Completed Successfully")


# # Autoencoder

# In[9]:


# =========================================
# STEP 4 : ADD CONTROLLED NOISE
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# NOISE FACTOR
# =========================================

noise_factor = 0.05

# =========================================
# ADD GAUSSIAN NOISE
# =========================================

noisy_images = processed_images + (
    noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=processed_images.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

noisy_images = np.clip(
    noisy_images,
    0.0,
    1.0
)

# =========================================
# DISPLAY SAMPLE
# =========================================

plt.figure(figsize=(10,5))

# Original

plt.subplot(1,2,1)

plt.imshow(processed_images[0])

plt.title("Original Image")

plt.axis("off")

# Noisy

plt.subplot(1,2,2)

plt.imshow(noisy_images[0])

plt.title("Noisy Image")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Original Dataset Shape :",
      processed_images.shape)

print("Noisy Dataset Shape :",
      noisy_images.shape)

print("\nNoise Factor :",
      noise_factor)

print("\nControlled Noise Added Successfully")


# In[10]:


# =========================================
# STEP 5 : BUILD AUTOENCODER
# =========================================

from tensorflow.keras import layers
from tensorflow.keras import models

# =========================================
# BUILD MODEL
# =========================================

def build_autoencoder():

    # -------------------------------------
    # INPUT LAYER
    # -------------------------------------

    input_img = layers.Input(
        shape=(128,128,3)
    )

    # =====================================
    # ENCODER
    # =====================================

    x = layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )(input_img)

    x = layers.MaxPooling2D(
        (2,2),
        padding='same'
    )(x)

    x = layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )(x)

    x = layers.MaxPooling2D(
        (2,2),
        padding='same'
    )(x)

    # =====================================
    # BOTTLENECK
    # =====================================

    x = layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )(x)

    # =====================================
    # DECODER
    # =====================================

    x = layers.UpSampling2D(
        (2,2)
    )(x)

    x = layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )(x)

    x = layers.UpSampling2D(
        (2,2)
    )(x)

    # =====================================
    # OUTPUT LAYER
    # =====================================

    output = layers.Conv2D(
        3,
        (3,3),
        activation='sigmoid',
        padding='same'
    )(x)

    # =====================================
    # CREATE MODEL
    # =====================================

    model = models.Model(
        input_img,
        output
    )

    return model

# =========================================
# CREATE AUTOENCODER
# =========================================

autoencoder = build_autoencoder()

print("Autoencoder Model Created Successfully")


# In[11]:


# =========================================
# STEP 6 : AUTOENCODER SUMMARY
# =========================================

autoencoder.summary()


# In[12]:


# =========================================
# STEP 7 : COMPILE AUTOENCODER
# =========================================

autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# =========================================
# DISPLAY DETAILS
# =========================================

print("Autoencoder Compiled Successfully\n")

print("Optimizer      : Adam")

print("Learning Rate  : 0.001")

print("Loss Function  : Mean Squared Error")

print("Metric         : Accuracy")


# In[13]:


# =========================================
# STEP 8 : TRAIN AUTOENCODER
# =========================================

epochs = 50

batch_size = 8

validation_split = 0.2

# =========================================
# DISPLAY PARAMETERS
# =========================================

print("Training Parameters")
print("--------------------------")

print("Epochs           :", epochs)

print("Batch Size       :", batch_size)

print("Validation Split :", validation_split)

# =========================================
# TRAIN MODEL
# =========================================

history_autoencoder = autoencoder.fit(
    noisy_images,
    processed_images,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    shuffle=True,
    verbose=1
)

# =========================================
# TRAINING COMPLETED
# =========================================

print("\nAutoencoder Training Completed Successfully")


# In[14]:


# =========================================
# STEP 9 : ACCURACY & LOSS GRAPH
# =========================================

import matplotlib.pyplot as plt

# =========================================
# ACCURACY GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
    history_autoencoder.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history_autoencoder.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Autoencoder Accuracy Graph")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# =========================================
# LOSS GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
    history_autoencoder.history['loss'],
    label='Training Loss'
)

plt.plot(
    history_autoencoder.history['val_loss'],
    label='Validation Loss'
)

plt.title("Autoencoder Loss Graph")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

print("Accuracy & Loss Graph Generated Successfully")


# In[15]:


# =========================================
# STEP 10 : GENERATE RECONSTRUCTION
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# CREATE NOISY TEST IMAGE
# =========================================

test_noise_factor = 0.05

test_noisy_image = test_image + (
    test_noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=test_image.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

test_noisy_image = np.clip(
    test_noisy_image,
    0.0,
    1.0
)

# =========================================
# ADD BATCH DIMENSION
# =========================================

test_noisy_input = np.expand_dims(
    test_noisy_image,
    axis=0
)

# =========================================
# GENERATE RECONSTRUCTION
# =========================================

reconstructed_output = autoencoder.predict(
    test_noisy_input
)

# =========================================
# REMOVE BATCH DIMENSION
# =========================================

reconstructed_image = reconstructed_output[0]

# =========================================
# DISPLAY RESULTS
# =========================================

plt.figure(figsize=(15,5))

# -----------------------------------------
# ORIGINAL IMAGE
# -----------------------------------------

plt.subplot(1,3,1)

plt.imshow(test_image)

plt.title("Original Image")

plt.axis("off")

# -----------------------------------------
# NOISY IMAGE
# -----------------------------------------

plt.subplot(1,3,2)

plt.imshow(test_noisy_image)

plt.title("Noisy Image")

plt.axis("off")

# -----------------------------------------
# RECONSTRUCTED IMAGE
# -----------------------------------------

plt.subplot(1,3,3)

plt.imshow(reconstructed_image)

plt.title("Reconstructed Image")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Test Noise Factor :",
      test_noise_factor)

print("\nReconstruction Generated Successfully")


# In[16]:


# =========================================
# STEP 11 : PERFORMANCE METRICS
# MSE + PSNR + SSIM + IoU + DICE
# =========================================

import cv2
import numpy as np

from sklearn.metrics import mean_squared_error

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# =========================================
# CONVERT TO UINT8
# =========================================

original_uint8 = (
    test_image * 255
).astype(np.uint8)

reconstructed_uint8 = (
    reconstructed_image * 255
).astype(np.uint8)

# =========================================
# MSE
# =========================================

mse_value = mean_squared_error(
    original_uint8.flatten(),
    reconstructed_uint8.flatten()
)

# =========================================
# PSNR
# =========================================

psnr_value = peak_signal_noise_ratio(
    original_uint8,
    reconstructed_uint8
)

# =========================================
# SSIM
# =========================================

ssim_value = structural_similarity(
    original_uint8,
    reconstructed_uint8,
    channel_axis=-1
)

# =========================================
# CONVERT TO GRAYSCALE
# =========================================

original_gray = cv2.cvtColor(
    original_uint8,
    cv2.COLOR_RGB2GRAY
)

reconstructed_gray = cv2.cvtColor(
    reconstructed_uint8,
    cv2.COLOR_RGB2GRAY
)

# =========================================
# BINARY THRESHOLD
# =========================================

_, original_binary = cv2.threshold(
    original_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

_, reconstructed_binary = cv2.threshold(
    reconstructed_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

# =========================================
# IoU
# =========================================

intersection = np.logical_and(
    original_binary,
    reconstructed_binary
)

union = np.logical_or(
    original_binary,
    reconstructed_binary
)

iou_value = np.sum(intersection) / np.sum(union)

# =========================================
# DICE COEFFICIENT
# =========================================

dice_value = (
    2 * np.sum(intersection)
) / (
    np.sum(original_binary) +
    np.sum(reconstructed_binary)
)

# =========================================
# DISPLAY RESULTS
# =========================================

print("===================================")
print("AUTOENCODER PERFORMANCE METRICS")
print("===================================\n")

print("MSE Value   :",
      round(mse_value,4))

print("PSNR Value  :",
      round(psnr_value,4))

print("SSIM Value  :",
      round(ssim_value,4))

print("IoU Value   :",
      round(iou_value,4))

print("Dice Value  :",
      round(dice_value,4))


# In[17]:


# =========================================
# STEP 12 : CONFUSION MATRIX
# =========================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# =========================================
# FLATTEN BINARY IMAGES
# =========================================

y_true = original_binary.flatten()

y_pred = reconstructed_binary.flatten()

# =========================================
# GENERATE CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

# =========================================
# DISPLAY MATRIX
# =========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap='Blues')

plt.title("Autoencoder Confusion Matrix")

plt.show()

print("Confusion Matrix Generated Successfully")


# In[18]:


# =========================================
# STEP 13 : SAVE RECONSTRUCTED IMAGE
# =========================================

import cv2

# =========================================
# RGB → BGR
# =========================================

save_image = cv2.cvtColor(
    reconstructed_uint8,
    cv2.COLOR_RGB2BGR
)

# =========================================
# SAVE IMAGE
# =========================================

save_path = r"C:\Users\money\OneDrive\Desktop\Autoencoder_Reconstructed.jpg"

cv2.imwrite(
    save_path,
    save_image
)

print("Reconstructed Image Saved Successfully")

print("Saved Location :")

print(save_path)


# In[19]:


# =========================================
# SAVE CONFUSION MATRIX ONLY
# =========================================

confusion_save_path = r"C:\Users\money\OneDrive\Desktop\Autoencoder_ConfusionMatrix.jpg"

plt.savefig(
    confusion_save_path,
    dpi=300,
    bbox_inches='tight'
)

print("Confusion Matrix Saved Successfully")

print("Saved Location :")

print(confusion_save_path)


# # DENOISING AUTOENCODER

# In[20]:


# =========================================
# DENOISING AUTOENCODER
# STEP 1 : CREATE DENOISING DATASET
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# NOISE FACTOR
# =========================================

noise_factor = 0.20

# =========================================
# ADD GAUSSIAN NOISE
# =========================================

denoise_noisy_images = processed_images + (
    noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=processed_images.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

denoise_noisy_images = np.clip(
    denoise_noisy_images,
    0.0,
    1.0
)

# =========================================
# DISPLAY SAMPLE
# =========================================

plt.figure(figsize=(10,5))

# Original

plt.subplot(1,2,1)

plt.imshow(processed_images[0])

plt.title("Original Image")

plt.axis("off")

# Noisy

plt.subplot(1,2,2)

plt.imshow(denoise_noisy_images[0])

plt.title("Noisy Image")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Original Dataset Shape :",
      processed_images.shape)

print("Noisy Dataset Shape :",
      denoise_noisy_images.shape)

print("\nNoise Factor :",
      noise_factor)

print("\nDenoising Dataset Created Successfully")


# In[21]:


# =========================================
# STEP 2 : BUILD DENOISING AUTOENCODER
# =========================================

from tensorflow.keras import layers
from tensorflow.keras import models

# =========================================
# BUILD MODEL
# =========================================

def build_denoising_autoencoder():

    # -------------------------------------
    # INPUT LAYER
    # -------------------------------------

    input_img = layers.Input(
        shape=(128,128,3)
    )

    # =====================================
    # ENCODER
    # =====================================

    x = layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )(input_img)

    x = layers.MaxPooling2D(
        (2,2),
        padding='same'
    )(x)

    x = layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )(x)

    x = layers.MaxPooling2D(
        (2,2),
        padding='same'
    )(x)

    # =====================================
    # BOTTLENECK
    # =====================================

    x = layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )(x)

    # =====================================
    # DECODER
    # =====================================

    x = layers.UpSampling2D(
        (2,2)
    )(x)

    x = layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )(x)

    x = layers.UpSampling2D(
        (2,2)
    )(x)

    # =====================================
    # OUTPUT LAYER
    # =====================================

    output = layers.Conv2D(
        3,
        (3,3),
        activation='sigmoid',
        padding='same'
    )(x)

    # =====================================
    # CREATE MODEL
    # =====================================

    model = models.Model(
        input_img,
        output
    )

    return model

# =========================================
# CREATE MODEL
# =========================================

denoising_autoencoder = build_denoising_autoencoder()

print("Denoising Autoencoder Created Successfully")


# In[22]:


# =========================================
# STEP 3 : MODEL SUMMARY
# =========================================

denoising_autoencoder.summary()


# In[23]:


# =========================================
# STEP 4 : COMPILE DENOISING AUTOENCODER
# =========================================

denoising_autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# =========================================
# DISPLAY DETAILS
# =========================================

print("Denoising Autoencoder Compiled Successfully\n")

print("Optimizer      : Adam")

print("Learning Rate  : 0.001")

print("Loss Function  : Mean Squared Error")

print("Metric         : Accuracy")


# In[24]:


# =========================================
# STEP 5 : TRAIN DENOISING AUTOENCODER
# =========================================

epochs = 50

batch_size = 8

validation_split = 0.2

# =========================================
# DISPLAY PARAMETERS
# =========================================

print("Training Parameters")
print("--------------------------")

print("Epochs           :", epochs)

print("Batch Size       :", batch_size)

print("Validation Split :", validation_split)

# =========================================
# TRAIN MODEL
# =========================================

history_denoising = denoising_autoencoder.fit(
    denoise_noisy_images,
    processed_images,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    shuffle=True,
    verbose=1
)

# =========================================
# TRAINING COMPLETED
# =========================================

print("\nDenoising Autoencoder Training Completed Successfully")


# In[25]:


# =========================================
# STEP 6 : ACCURACY & LOSS GRAPH
# =========================================

import matplotlib.pyplot as plt

# =========================================
# ACCURACY GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
    history_denoising.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history_denoising.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Denoising Autoencoder Accuracy Graph")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# =========================================
# LOSS GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
    history_denoising.history['loss'],
    label='Training Loss'
)

plt.plot(
    history_denoising.history['val_loss'],
    label='Validation Loss'
)

plt.title("Denoising Autoencoder Loss Graph")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

print("Accuracy & Loss Graph Generated Successfully")


# In[26]:


# =========================================
# STEP 7 : GENERATE DENOISING RECONSTRUCTION
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# CREATE NOISY TEST IMAGE
# =========================================

test_noise_factor = 0.20

test_noisy_image = test_image + (
    test_noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=test_image.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

test_noisy_image = np.clip(
    test_noisy_image,
    0.0,
    1.0
)

# =========================================
# ADD BATCH DIMENSION
# =========================================

test_noisy_input = np.expand_dims(
    test_noisy_image,
    axis=0
)

# =========================================
# GENERATE RECONSTRUCTION
# =========================================

denoise_output = denoising_autoencoder.predict(
    test_noisy_input
)

# =========================================
# REMOVE BATCH DIMENSION
# =========================================

denoise_reconstructed = denoise_output[0]

# =========================================
# DISPLAY RESULTS
# =========================================

plt.figure(figsize=(15,5))

# Original

plt.subplot(1,3,1)

plt.imshow(test_image)

plt.title("Original Image")

plt.axis("off")

# Noisy

plt.subplot(1,3,2)

plt.imshow(test_noisy_image)

plt.title("Noisy Image")

plt.axis("off")

# Reconstructed

plt.subplot(1,3,3)

plt.imshow(denoise_reconstructed)

plt.title("Denoised Reconstruction")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Test Noise Factor :",
      test_noise_factor)

print("\nDenoising Reconstruction Generated Successfully")


# In[27]:


# =========================================
# STEP 8 : PERFORMANCE METRICS
# MSE + PSNR + SSIM + IoU + DICE
# =========================================

import cv2
import numpy as np

from sklearn.metrics import mean_squared_error

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# =========================================
# CONVERT TO UINT8
# =========================================

original_uint8 = (
    test_image * 255
).astype(np.uint8)

denoise_uint8 = (
    denoise_reconstructed * 255
).astype(np.uint8)

# =========================================
# MSE
# =========================================

mse_value = mean_squared_error(
    original_uint8.flatten(),
    denoise_uint8.flatten()
)

# =========================================
# PSNR
# =========================================

psnr_value = peak_signal_noise_ratio(
    original_uint8,
    denoise_uint8
)

# =========================================
# SSIM
# =========================================

ssim_value = structural_similarity(
    original_uint8,
    denoise_uint8,
    channel_axis=-1
)

# =========================================
# CONVERT TO GRAYSCALE
# =========================================

original_gray = cv2.cvtColor(
    original_uint8,
    cv2.COLOR_RGB2GRAY
)

denoise_gray = cv2.cvtColor(
    denoise_uint8,
    cv2.COLOR_RGB2GRAY
)

# =========================================
# BINARY THRESHOLD
# =========================================

_, original_binary = cv2.threshold(
    original_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

_, denoise_binary = cv2.threshold(
    denoise_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

# =========================================
# IoU
# =========================================

intersection = np.logical_and(
    original_binary,
    denoise_binary
)

union = np.logical_or(
    original_binary,
    denoise_binary
)

iou_value = np.sum(intersection) / np.sum(union)

# =========================================
# DICE COEFFICIENT
# =========================================

dice_value = (
    2 * np.sum(intersection)
) / (
    np.sum(original_binary) +
    np.sum(denoise_binary)
)

# =========================================
# DISPLAY RESULTS
# =========================================

print("===================================")
print("DENOISING AUTOENCODER METRICS")
print("===================================\n")

print("MSE Value   :",
      round(mse_value,4))

print("PSNR Value  :",
      round(psnr_value,4))

print("SSIM Value  :",
      round(ssim_value,4))

print("IoU Value   :",
      round(iou_value,4))

print("Dice Value  :",
      round(dice_value,4))


# In[28]:


# =========================================
# STEP 9 : CONFUSION MATRIX
# =========================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# =========================================
# FLATTEN BINARY IMAGES
# =========================================

y_true = original_binary.flatten()

y_pred = denoise_binary.flatten()

# =========================================
# GENERATE CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

# =========================================
# DISPLAY MATRIX
# =========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap='Blues')

plt.title("Denoising Autoencoder Confusion Matrix")

plt.show()

print("Confusion Matrix Generated Successfully")


# In[29]:


# =========================================
# STEP 10 : SAVE RECONSTRUCTED IMAGE
# =========================================

import cv2

# =========================================
# RGB → BGR
# =========================================

save_image = cv2.cvtColor(
    denoise_uint8,
    cv2.COLOR_RGB2BGR
)

# =========================================
# SAVE IMAGE
# =========================================

save_path = r"C:\Users\money\OneDrive\Desktop\DenoisingAE_Reconstructed.jpg"

cv2.imwrite(
    save_path,
    save_image
)

print("Reconstructed Image Saved Successfully")

print("Saved Location :")

print(save_path)


# In[30]:


# =========================================
# STEP 11 : SAVE CONFUSION MATRIX
# =========================================

confusion_save_path = r"C:\Users\money\OneDrive\Desktop\DenoisingAE_ConfusionMatrix.jpg"

plt.savefig(
    confusion_save_path,
    dpi=300,
    bbox_inches='tight'
)

print("Confusion Matrix Saved Successfully")

print("Saved Location :")

print(confusion_save_path)


# # BUILD U-NET

# In[31]:


# =========================================
# U-NET MODEL
# STEP 1 : BUILD U-NET
# =========================================

from tensorflow.keras import layers
from tensorflow.keras import models

# =========================================
# BUILD U-NET
# =========================================

def build_unet():

    inputs = layers.Input(
        shape=(128,128,3)
    )

    # =====================================
    # ENCODER
    # =====================================

    c1 = layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )(inputs)

    c1 = layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )(c1)

    p1 = layers.MaxPooling2D(
        (2,2)
    )(c1)

    # -------------------------------------

    c2 = layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )(p1)

    c2 = layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )(c2)

    p2 = layers.MaxPooling2D(
        (2,2)
    )(c2)

    # =====================================
    # BOTTLENECK
    # =====================================

    bn = layers.Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    )(p2)

    bn = layers.Conv2D(
        128,
        (3,3),
        activation='relu',
        padding='same'
    )(bn)

    # =====================================
    # DECODER
    # =====================================

    u1 = layers.UpSampling2D(
        (2,2)
    )(bn)

    u1 = layers.concatenate(
        [u1, c2]
    )

    c3 = layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )(u1)

    c3 = layers.Conv2D(
        64,
        (3,3),
        activation='relu',
        padding='same'
    )(c3)

    # -------------------------------------

    u2 = layers.UpSampling2D(
        (2,2)
    )(c3)

    u2 = layers.concatenate(
        [u2, c1]
    )

    c4 = layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )(u2)

    c4 = layers.Conv2D(
        32,
        (3,3),
        activation='relu',
        padding='same'
    )(c4)

    # =====================================
    # OUTPUT LAYER
    # =====================================

    outputs = layers.Conv2D(
        3,
        (1,1),
        activation='sigmoid'
    )(c4)

    # =====================================
    # CREATE MODEL
    # =====================================

    model = models.Model(
        inputs,
        outputs
    )

    return model

# =========================================
# CREATE MODEL
# =========================================

unet_model = build_unet()

print("U-Net Model Created Successfully")


# In[32]:


# =========================================
# STEP 2 : CREATE NOISY DATASET FOR U-NET
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# NOISE FACTOR
# =========================================

unet_noise_factor = 0.05

# =========================================
# ADD GAUSSIAN NOISE
# =========================================

unet_noisy_images = processed_images + (
    unet_noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=processed_images.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

unet_noisy_images = np.clip(
    unet_noisy_images,
    0.0,
    1.0
)

# =========================================
# DISPLAY SAMPLE
# =========================================

plt.figure(figsize=(10,5))

# Original

plt.subplot(1,2,1)

plt.imshow(processed_images[0])

plt.title("Original Image")

plt.axis("off")

# Noisy

plt.subplot(1,2,2)

plt.imshow(unet_noisy_images[0])

plt.title("Noisy Image")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Noise Factor :",
      unet_noise_factor)

print("\nU-Net Noisy Dataset Created Successfully")


# In[33]:


# =========================================
# STEP 3 : MODEL SUMMARY
# =========================================

unet_model.summary()


# In[34]:


# =========================================
# STEP 4 : COMPILE U-NET
# =========================================

unet_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# =========================================
# DISPLAY DETAILS
# =========================================

print("U-Net Compiled Successfully\n")

print("Optimizer      : Adam")

print("Learning Rate  : 0.001")

print("Loss Function  : Mean Squared Error")

print("Metric         : Accuracy")


# In[35]:


# =========================================
# STEP 5 : TRAIN U-NET
# =========================================

epochs = 50

batch_size = 8

validation_split = 0.2

# =========================================
# DISPLAY PARAMETERS
# =========================================

print("Training Parameters")
print("--------------------------")

print("Epochs           :", epochs)

print("Batch Size       :", batch_size)

print("Validation Split :", validation_split)

# =========================================
# TRAIN MODEL
# =========================================

history_unet = unet_model.fit(
    unet_noisy_images,
    processed_images,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    shuffle=True,
    verbose=1
)

# =========================================
# TRAINING COMPLETED
# =========================================

print("\nU-Net Training Completed Successfully")


# In[36]:


# =========================================
# STEP 6 : ACCURACY & LOSS GRAPH
# =========================================

import matplotlib.pyplot as plt

# =========================================
# ACCURACY GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
    history_unet.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history_unet.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("U-Net Accuracy Graph")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# =========================================
# LOSS GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
    history_unet.history['loss'],
    label='Training Loss'
)

plt.plot(
    history_unet.history['val_loss'],
    label='Validation Loss'
)

plt.title("U-Net Loss Graph")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

print("Accuracy & Loss Graph Generated Successfully")


# In[37]:


# =========================================
# STEP 7 : GENERATE U-NET RECONSTRUCTION
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# CREATE NOISY TEST IMAGE
# =========================================

test_noise_factor = 0.05

test_noisy_image = test_image + (
    test_noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=test_image.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

test_noisy_image = np.clip(
    test_noisy_image,
    0.0,
    1.0
)

# =========================================
# ADD BATCH DIMENSION
# =========================================

test_noisy_input = np.expand_dims(
    test_noisy_image,
    axis=0
)

# =========================================
# GENERATE RECONSTRUCTION
# =========================================

unet_output = unet_model.predict(
    test_noisy_input
)

# =========================================
# REMOVE BATCH DIMENSION
# =========================================

unet_reconstructed = unet_output[0]

# =========================================
# DISPLAY RESULTS
# =========================================

plt.figure(figsize=(15,5))

# Original

plt.subplot(1,3,1)

plt.imshow(test_image)

plt.title("Original Image")

plt.axis("off")

# Noisy

plt.subplot(1,3,2)

plt.imshow(test_noisy_image)

plt.title("Noisy Image")

plt.axis("off")

# Reconstructed

plt.subplot(1,3,3)

plt.imshow(unet_reconstructed)

plt.title("U-Net Reconstruction")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Test Noise Factor :",
      test_noise_factor)

print("\nU-Net Reconstruction Generated Successfully")


# In[38]:


# =========================================
# STEP 8 : PERFORMANCE METRICS
# MSE + PSNR + SSIM + IoU + DICE
# =========================================

import cv2
import numpy as np

from sklearn.metrics import mean_squared_error

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# =========================================
# CONVERT TO UINT8
# =========================================

original_uint8 = (
    test_image * 255
).astype(np.uint8)

unet_uint8 = (
    unet_reconstructed * 255
).astype(np.uint8)

# =========================================
# MSE
# =========================================

mse_value = mean_squared_error(
    original_uint8.flatten(),
    unet_uint8.flatten()
)

# =========================================
# PSNR
# =========================================

psnr_value = peak_signal_noise_ratio(
    original_uint8,
    unet_uint8
)

# =========================================
# SSIM
# =========================================

ssim_value = structural_similarity(
    original_uint8,
    unet_uint8,
    channel_axis=-1
)

# =========================================
# CONVERT TO GRAYSCALE
# =========================================

original_gray = cv2.cvtColor(
    original_uint8,
    cv2.COLOR_RGB2GRAY
)

unet_gray = cv2.cvtColor(
    unet_uint8,
    cv2.COLOR_RGB2GRAY
)

# =========================================
# BINARY THRESHOLD
# =========================================

_, original_binary = cv2.threshold(
    original_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

_, unet_binary = cv2.threshold(
    unet_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

# =========================================
# IoU
# =========================================

intersection = np.logical_and(
    original_binary,
    unet_binary
)

union = np.logical_or(
    original_binary,
    unet_binary
)

iou_value = np.sum(intersection) / np.sum(union)

# =========================================
# DICE COEFFICIENT
# =========================================

dice_value = (
    2 * np.sum(intersection)
) / (
    np.sum(original_binary) +
    np.sum(unet_binary)
)

# =========================================
# DISPLAY RESULTS
# =========================================

print("===================================")
print("U-NET PERFORMANCE METRICS")
print("===================================\n")

print("MSE Value   :",
      round(mse_value,4))

print("PSNR Value  :",
      round(psnr_value,4))

print("SSIM Value  :",
      round(ssim_value,4))

print("IoU Value   :",
      round(iou_value,4))

print("Dice Value  :",
      round(dice_value,4))


# In[39]:


# =========================================
# STEP 9 : CONFUSION MATRIX
# =========================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# =========================================
# FLATTEN BINARY IMAGES
# =========================================

y_true = original_binary.flatten()

y_pred = unet_binary.flatten()

# =========================================
# GENERATE CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

# =========================================
# DISPLAY MATRIX
# =========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap='Blues')

plt.title("U-Net Confusion Matrix")

plt.show()

print("Confusion Matrix Generated Successfully")


# In[40]:


# =========================================
# STEP 10 : SAVE RECONSTRUCTED IMAGE
# =========================================

import cv2

# =========================================
# RGB → BGR
# =========================================

save_image = cv2.cvtColor(
    unet_uint8,
    cv2.COLOR_RGB2BGR
)

# =========================================
# SAVE IMAGE
# =========================================

save_path = r"C:\Users\money\OneDrive\Desktop\UNet_Reconstructed.jpg"

cv2.imwrite(
    save_path,
    save_image
)

print("Reconstructed Image Saved Successfully")

print("Saved Location :")

print(save_path)


# In[41]:


# =========================================
# STEP 11 : SAVE CONFUSION MATRIX
# =========================================

confusion_save_path = r"C:\Users\money\OneDrive\Desktop\UNet_ConfusionMatrix.jpg"

plt.savefig(
    confusion_save_path,
    dpi=300,
    bbox_inches='tight'
)

print("Confusion Matrix Saved Successfully")

print("Saved Location :")

print(confusion_save_path)


# # SIMPLIFIED GAN MODEL

# In[42]:


# =========================================
# SIMPLIFIED GAN
# STEP 1 : CREATE NOISY DATASET
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# NOISE FACTOR
# =========================================

gan_noise_factor = 0.10

# =========================================
# ADD GAUSSIAN NOISE
# =========================================

gan_noisy_images = processed_images + (
    gan_noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=processed_images.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

gan_noisy_images = np.clip(
    gan_noisy_images,
    0.0,
    1.0
)

# =========================================
# DISPLAY SAMPLE
# =========================================

plt.figure(figsize=(10,5))

# Original

plt.subplot(1,2,1)

plt.imshow(processed_images[0])

plt.title("Original Image")

plt.axis("off")

# Noisy

plt.subplot(1,2,2)

plt.imshow(gan_noisy_images[0])

plt.title("GAN Noisy Image")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Noise Factor :",
      gan_noise_factor)

print("\nGAN Noisy Dataset Created Successfully")


# In[43]:


# =========================================
# STEP 2 : BUILD SIMPLIFIED GAN
# =========================================

from tensorflow.keras import layers
from tensorflow.keras import models

# =========================================
# BUILD GENERATOR
# =========================================

def build_generator():

    # -------------------------------------
    # INPUT LAYER
    # -------------------------------------

    inputs = layers.Input(
        shape=(128,128,3)
    )

    # =====================================
    # CONVOLUTION BLOCK 1
    # =====================================

    x = layers.Conv2D(
        64,
        (3,3),
        padding='same',
        activation='relu'
    )(inputs)

    # =====================================
    # CONVOLUTION BLOCK 2
    # =====================================

    x = layers.Conv2D(
        64,
        (3,3),
        padding='same',
        activation='relu'
    )(x)

    # =====================================
    # OUTPUT LAYER
    # =====================================

    outputs = layers.Conv2D(
        3,
        (3,3),
        padding='same',
        activation='sigmoid'
    )(x)

    # =====================================
    # CREATE MODEL
    # =====================================

    model = models.Model(
        inputs,
        outputs
    )

    return model

# =========================================
# CREATE MODEL
# =========================================

generator = build_generator()

print("Simplified GAN Model Created Successfully")


# In[44]:


# =========================================
# STEP 3 : MODEL SUMMARY
# =========================================

generator.summary()


# In[45]:


# =========================================
# STEP 4 : COMPILE SIMPLIFIED GAN
# =========================================

generator.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# =========================================
# DISPLAY DETAILS
# =========================================

print("Simplified GAN Compiled Successfully\n")

print("Optimizer      : Adam")

print("Learning Rate  : 0.001")

print("Loss Function  : Mean Squared Error")

print("Metric         : Accuracy")


# In[46]:


# =========================================
# STEP 5 : TRAIN SIMPLIFIED GAN
# =========================================

epochs = 50

batch_size = 8

validation_split = 0.2

# =========================================
# DISPLAY PARAMETERS
# =========================================

print("Training Parameters")
print("--------------------------")

print("Epochs           :", epochs)

print("Batch Size       :", batch_size)

print("Validation Split :", validation_split)

# =========================================
# TRAIN MODEL
# =========================================

history_gan = generator.fit(
    gan_noisy_images,
    processed_images,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    shuffle=True,
    verbose=1
)

# =========================================
# TRAINING COMPLETED
# =========================================

print("\nSimplified GAN Training Completed Successfully")


# In[47]:


# =========================================
# STEP 6 : ACCURACY & LOSS GRAPH
# =========================================

import matplotlib.pyplot as plt

# =========================================
# ACCURACY GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
    history_gan.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history_gan.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Simplified GAN Accuracy Graph")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# =========================================
# LOSS GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
    history_gan.history['loss'],
    label='Training Loss'
)

plt.plot(
    history_gan.history['val_loss'],
    label='Validation Loss'
)

plt.title("Simplified GAN Loss Graph")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

print("Accuracy & Loss Graph Generated Successfully")


# In[48]:


# =========================================
# STEP 7 : GENERATE GAN RECONSTRUCTION
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# CREATE NOISY TEST IMAGE
# =========================================

test_noise_factor = 0.10

test_noisy_image = test_image + (
    test_noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=test_image.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

test_noisy_image = np.clip(
    test_noisy_image,
    0.0,
    1.0
)

# =========================================
# ADD BATCH DIMENSION
# =========================================

test_noisy_input = np.expand_dims(
    test_noisy_image,
    axis=0
)

# =========================================
# GENERATE RECONSTRUCTION
# =========================================

gan_output = generator.predict(
    test_noisy_input
)

# =========================================
# REMOVE BATCH DIMENSION
# =========================================

gan_reconstructed = gan_output[0]

# =========================================
# DISPLAY RESULTS
# =========================================

plt.figure(figsize=(15,5))

# Original

plt.subplot(1,3,1)

plt.imshow(test_image)

plt.title("Original Image")

plt.axis("off")

# Noisy

plt.subplot(1,3,2)

plt.imshow(test_noisy_image)

plt.title("Noisy Image")

plt.axis("off")

# Reconstructed

plt.subplot(1,3,3)

plt.imshow(gan_reconstructed)

plt.title("Simplified GAN Reconstruction")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Test Noise Factor :",
      test_noise_factor)

print("\nGAN Reconstruction Generated Successfully")


# In[49]:


# =========================================
# STEP 8 : PERFORMANCE METRICS
# MSE + PSNR + SSIM + IoU + DICE
# =========================================

import cv2
import numpy as np

from sklearn.metrics import mean_squared_error

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# =========================================
# CONVERT TO UINT8
# =========================================

original_uint8 = (
    test_image * 255
).astype(np.uint8)

gan_uint8 = (
    gan_reconstructed * 255
).astype(np.uint8)

# =========================================
# MSE
# =========================================

mse_value = mean_squared_error(
    original_uint8.flatten(),
    gan_uint8.flatten()
)

# =========================================
# PSNR
# =========================================

psnr_value = peak_signal_noise_ratio(
    original_uint8,
    gan_uint8
)

# =========================================
# SSIM
# =========================================

ssim_value = structural_similarity(
    original_uint8,
    gan_uint8,
    channel_axis=-1
)

# =========================================
# CONVERT TO GRAYSCALE
# =========================================

original_gray = cv2.cvtColor(
    original_uint8,
    cv2.COLOR_RGB2GRAY
)

gan_gray = cv2.cvtColor(
    gan_uint8,
    cv2.COLOR_RGB2GRAY
)

# =========================================
# BINARY THRESHOLD
# =========================================

_, original_binary = cv2.threshold(
    original_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

_, gan_binary = cv2.threshold(
    gan_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

# =========================================
# IoU
# =========================================

intersection = np.logical_and(
    original_binary,
    gan_binary
)

union = np.logical_or(
    original_binary,
    gan_binary
)

iou_value = np.sum(intersection) / np.sum(union)

# =========================================
# DICE COEFFICIENT
# =========================================

dice_value = (
    2 * np.sum(intersection)
) / (
    np.sum(original_binary) +
    np.sum(gan_binary)
)

# =========================================
# DISPLAY RESULTS
# =========================================

print("===================================")
print("SIMPLIFIED GAN METRICS")
print("===================================\n")

print("MSE Value   :",
      round(mse_value,4))

print("PSNR Value  :",
      round(psnr_value,4))

print("SSIM Value  :",
      round(ssim_value,4))

print("IoU Value   :",
      round(iou_value,4))

print("Dice Value  :",
      round(dice_value,4))


# In[50]:


# =========================================
# STEP 9 : CONFUSION MATRIX
# =========================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# =========================================
# FLATTEN BINARY IMAGES
# =========================================

y_true = original_binary.flatten()

y_pred = gan_binary.flatten()

# =========================================
# GENERATE CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

# =========================================
# DISPLAY MATRIX
# =========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap='Blues')

plt.title("Simplified GAN Confusion Matrix")

plt.show()

print("Confusion Matrix Generated Successfully")


# In[51]:


# =========================================
# STEP 10 : SAVE RECONSTRUCTED IMAGE
# =========================================

import cv2

# =========================================
# RGB → BGR
# =========================================

save_image = cv2.cvtColor(
    gan_uint8,
    cv2.COLOR_RGB2BGR
)

# =========================================
# SAVE IMAGE
# =========================================

save_path = r"C:\Users\money\OneDrive\Desktop\SimplifiedGAN_Reconstructed.jpg"

cv2.imwrite(
    save_path,
    save_image
)

print("Reconstructed Image Saved Successfully")

print("Saved Location :")

print(save_path)


# In[52]:


# =========================================
# STEP 11 : SAVE CONFUSION MATRIX
# =========================================

confusion_save_path = r"C:\Users\money\OneDrive\Desktop\SimplifiedGAN_ConfusionMatrix.jpg"

plt.savefig(
    confusion_save_path,
    dpi=300,
    bbox_inches='tight'
)

print("Confusion Matrix Saved Successfully")

print("Saved Location :")

print(confusion_save_path)


# # PIX2PIX MODEL

# In[53]:


# =========================================
# PIX2PIX MODEL
# STEP 1 : CREATE NOISY DATASET
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# NOISE FACTOR
# =========================================

pix2pix_noise_factor = 0.10

# =========================================
# ADD GAUSSIAN NOISE
# =========================================

pix2pix_noisy_images = processed_images + (
    pix2pix_noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=processed_images.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

pix2pix_noisy_images = np.clip(
    pix2pix_noisy_images,
    0.0,
    1.0
)

# =========================================
# DISPLAY SAMPLE
# =========================================

plt.figure(figsize=(10,5))

# Original

plt.subplot(1,2,1)

plt.imshow(processed_images[0])

plt.title("Original Image")

plt.axis("off")

# Noisy

plt.subplot(1,2,2)

plt.imshow(pix2pix_noisy_images[0])

plt.title("Pix2Pix Noisy Image")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Noise Factor :",
      pix2pix_noise_factor)

print("\nPix2Pix Noisy Dataset Created Successfully")


# In[55]:


# =========================================
# STEP 2 : BUILD PIX2PIX GENERATOR
# FIXED VERSION
# =========================================

from tensorflow.keras import layers
from tensorflow.keras import models

# =========================================
# BUILD GENERATOR
# =========================================

def build_pix2pix_generator():

    # -------------------------------------
    # INPUT
    # -------------------------------------

    inputs = layers.Input(
        shape=(128,128,3)
    )

    # =====================================
    # ENCODER
    # =====================================

    e1 = layers.Conv2D(
        64,
        (4,4),
        strides=2,
        padding='same',
        activation='relu'
    )(inputs)
    # 64x64

    e2 = layers.Conv2D(
        128,
        (4,4),
        strides=2,
        padding='same',
        activation='relu'
    )(e1)
    # 32x32

    # =====================================
    # BOTTLENECK
    # =====================================

    b = layers.Conv2D(
        256,
        (4,4),
        padding='same',
        activation='relu'
    )(e2)
    # 32x32

    # =====================================
    # DECODER
    # =====================================

    # -------------------------------------
    # FIRST UPSAMPLING
    # -------------------------------------

    d1 = layers.UpSampling2D(
        (2,2)
    )(b)
    # 64x64

    d1 = layers.Conv2D(
        128,
        (3,3),
        padding='same',
        activation='relu'
    )(d1)

    # CONCAT WITH e1 (64x64)
    d1 = layers.concatenate([d1, e1])

    # -------------------------------------
    # SECOND UPSAMPLING
    # -------------------------------------

    d2 = layers.Conv2D(
        64,
        (3,3),
        padding='same',
        activation='relu'
    )(d1)

    d2 = layers.UpSampling2D(
        (2,2)
    )(d2)
    # 128x128

    # =====================================
    # OUTPUT
    # =====================================

    outputs = layers.Conv2D(
        3,
        (1,1),
        activation='sigmoid'
    )(d2)

    # =====================================
    # CREATE MODEL
    # =====================================

    model = models.Model(
        inputs,
        outputs
    )

    return model

# =========================================
# CREATE GENERATOR
# =========================================

pix2pix_generator = build_pix2pix_generator()

print("Pix2Pix Generator Created Successfully")


# In[56]:


# =========================================
# STEP 3 : BUILD PIX2PIX DISCRIMINATOR
# =========================================

from tensorflow.keras import layers
from tensorflow.keras import models

# =========================================
# BUILD DISCRIMINATOR
# =========================================

def build_discriminator():

    # -------------------------------------
    # INPUT IMAGE
    # -------------------------------------

    input_image = layers.Input(
        shape=(128,128,3)
    )

    # -------------------------------------
    # TARGET IMAGE
    # -------------------------------------

    target_image = layers.Input(
        shape=(128,128,3)
    )

    # =====================================
    # CONCAT INPUT + TARGET
    # =====================================

    merged = layers.concatenate(
        [input_image, target_image]
    )

    # =====================================
    # CONVOLUTION BLOCKS
    # =====================================

    x = layers.Conv2D(
        64,
        (4,4),
        strides=2,
        padding='same',
        activation='relu'
    )(merged)

    x = layers.Conv2D(
        128,
        (4,4),
        strides=2,
        padding='same',
        activation='relu'
    )(x)

    x = layers.Conv2D(
        256,
        (4,4),
        padding='same',
        activation='relu'
    )(x)

    # =====================================
    # OUTPUT LAYER
    # =====================================

    outputs = layers.Conv2D(
        1,
        (4,4),
        padding='same',
        activation='sigmoid'
    )(x)

    # =====================================
    # CREATE MODEL
    # =====================================

    model = models.Model(
        [input_image, target_image],
        outputs
    )

    return model

# =========================================
# CREATE DISCRIMINATOR
# =========================================

pix2pix_discriminator = build_discriminator()

print("Pix2Pix Discriminator Created Successfully")


# In[57]:


# =========================================
# STEP 4 : BUILD PIX2PIX GAN
# =========================================

from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam

# =========================================
# COMPILE DISCRIMINATOR
# =========================================

pix2pix_discriminator.compile(
    optimizer=Adam(learning_rate=0.0002),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================================
# FREEZE DISCRIMINATOR
# =========================================

pix2pix_discriminator.trainable = False

# =========================================
# INPUT IMAGE
# =========================================

input_image = layers.Input(
    shape=(128,128,3)
)

# =========================================
# GENERATOR OUTPUT
# =========================================

generated_image = pix2pix_generator(
    input_image
)

# =========================================
# DISCRIMINATOR OUTPUT
# =========================================

validity = pix2pix_discriminator(
    [input_image, generated_image]
)

# =========================================
# CREATE PIX2PIX GAN
# =========================================

pix2pix_gan = models.Model(
    input_image,
    [generated_image, validity]
)

# =========================================
# COMPILE GAN
# =========================================

pix2pix_gan.compile(
    optimizer=Adam(learning_rate=0.0002),
    loss=['mse', 'binary_crossentropy'],
    loss_weights=[100,1],
    metrics=['accuracy']
)

print("Pix2Pix GAN Created Successfully")


# In[59]:


# =========================================
# COMPILE PIX2PIX GENERATOR
# =========================================

pix2pix_generator.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

print("Pix2Pix Generator Compiled Successfully")


# In[60]:


history_pix2pix = pix2pix_generator.fit(
    pix2pix_noisy_images,
    processed_images,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)


# In[61]:


# =========================================
# STEP 6 : ACCURACY & LOSS GRAPH
# =========================================

import matplotlib.pyplot as plt

# =========================================
# ACCURACY GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
   history_pix2pix.history['accuracy'],
   label='Training Accuracy'
)

plt.plot(
   history_pix2pix.history['val_accuracy'],
   label='Validation Accuracy'
)

plt.title("Pix2Pix Accuracy Graph")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# =========================================
# LOSS GRAPH
# =========================================

plt.figure(figsize=(8,5))

plt.plot(
   history_pix2pix.history['loss'],
   label='Training Loss'
)

plt.plot(
   history_pix2pix.history['val_loss'],
   label='Validation Loss'
)

plt.title("Pix2Pix Loss Graph")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

print("Accuracy & Loss Graph Generated Successfully")
   


# In[62]:


pix2pix_generator.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)


# In[63]:


# =========================================
# STEP 7 : GENERATE PIX2PIX RECONSTRUCTION
# =========================================

import numpy as np
import matplotlib.pyplot as plt

# =========================================
# CREATE NOISY TEST IMAGE
# =========================================

test_noise_factor = 0.10

test_noisy_image = test_image + (
    test_noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=test_image.shape
    )
)

# =========================================
# CLIP VALUES
# =========================================

test_noisy_image = np.clip(
    test_noisy_image,
    0.0,
    1.0
)

# =========================================
# ADD BATCH DIMENSION
# =========================================

test_noisy_input = np.expand_dims(
    test_noisy_image,
    axis=0
)

# =========================================
# GENERATE RECONSTRUCTION
# =========================================

pix2pix_output = pix2pix_generator.predict(
    test_noisy_input
)

# =========================================
# REMOVE BATCH DIMENSION
# =========================================

pix2pix_reconstructed = pix2pix_output[0]

# =========================================
# DISPLAY RESULTS
# =========================================

plt.figure(figsize=(15,5))

# Original

plt.subplot(1,3,1)

plt.imshow(test_image)

plt.title("Original Image")

plt.axis("off")

# Noisy

plt.subplot(1,3,2)

plt.imshow(test_noisy_image)

plt.title("Noisy Image")

plt.axis("off")

# Reconstructed

plt.subplot(1,3,3)

plt.imshow(pix2pix_reconstructed)

plt.title("Pix2Pix Reconstruction")

plt.axis("off")

plt.show()

# =========================================
# DISPLAY DETAILS
# =========================================

print("Test Noise Factor :",
      test_noise_factor)

print("\nPix2Pix Reconstruction Generated Successfully")


# In[64]:


# =========================================
# STEP 8 : PERFORMANCE METRICS
# MSE + PSNR + SSIM + IoU + DICE
# =========================================

import cv2
import numpy as np

from sklearn.metrics import mean_squared_error

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

# =========================================
# CONVERT TO UINT8
# =========================================

original_uint8 = (
    test_image * 255
).astype(np.uint8)

pix2pix_uint8 = (
    pix2pix_reconstructed * 255
).astype(np.uint8)

# =========================================
# MSE
# =========================================

mse_value = mean_squared_error(
    original_uint8.flatten(),
    pix2pix_uint8.flatten()
)

# =========================================
# PSNR
# =========================================

psnr_value = peak_signal_noise_ratio(
    original_uint8,
    pix2pix_uint8
)

# =========================================
# SSIM
# =========================================

ssim_value = structural_similarity(
    original_uint8,
    pix2pix_uint8,
    channel_axis=-1
)

# =========================================
# CONVERT TO GRAYSCALE
# =========================================

original_gray = cv2.cvtColor(
    original_uint8,
    cv2.COLOR_RGB2GRAY
)

pix2pix_gray = cv2.cvtColor(
    pix2pix_uint8,
    cv2.COLOR_RGB2GRAY
)

# =========================================
# BINARY THRESHOLD
# =========================================

_, original_binary = cv2.threshold(
    original_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

_, pix2pix_binary = cv2.threshold(
    pix2pix_gray,
    127,
    1,
    cv2.THRESH_BINARY
)

# =========================================
# IoU
# =========================================

intersection = np.logical_and(
    original_binary,
    pix2pix_binary
)

union = np.logical_or(
    original_binary,
    pix2pix_binary
)

iou_value = np.sum(intersection) / np.sum(union)

# =========================================
# DICE COEFFICIENT
# =========================================

dice_value = (
    2 * np.sum(intersection)
) / (
    np.sum(original_binary) +
    np.sum(pix2pix_binary)
)

# =========================================
# DISPLAY RESULTS
# =========================================

print("===================================")
print("PIX2PIX PERFORMANCE METRICS")
print("===================================\n")

print("MSE Value   :",
      round(mse_value,4))

print("PSNR Value  :",
      round(psnr_value,4))

print("SSIM Value  :",
      round(ssim_value,4))

print("IoU Value   :",
      round(iou_value,4))

print("Dice Value  :",
      round(dice_value,4))


# In[65]:


# =========================================
# STEP 9 : CONFUSION MATRIX
# =========================================

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# =========================================
# FLATTEN BINARY IMAGES
# =========================================

y_true = original_binary.flatten()

y_pred = pix2pix_binary.flatten()

# =========================================
# GENERATE CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

# =========================================
# DISPLAY MATRIX
# =========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap='Blues')

plt.title("Pix2Pix Confusion Matrix")

plt.show()

print("Confusion Matrix Generated Successfully")


# In[66]:


# =========================================
# STEP 10 : SAVE RECONSTRUCTED IMAGE
# =========================================

import cv2

# =========================================
# RGB → BGR
# =========================================

save_image = cv2.cvtColor(
    pix2pix_uint8,
    cv2.COLOR_RGB2BGR
)

# =========================================
# SAVE IMAGE
# =========================================

save_path = r"C:\Users\money\OneDrive\Desktop\Pix2Pix_Reconstructed.jpg"

cv2.imwrite(
    save_path,
    save_image
)

print("Reconstructed Image Saved Successfully")

print("Saved Location :")

print(save_path)


# In[67]:


# =========================================
# STEP 11 : SAVE CONFUSION MATRIX
# =========================================

confusion_save_path = r"C:\Users\money\OneDrive\Desktop\Pix2Pix_ConfusionMatrix.jpg"

plt.savefig(
    confusion_save_path,
    dpi=300,
    bbox_inches='tight'
)

print("Confusion Matrix Saved Successfully")

print("Saved Location :")

print(confusion_save_path)


# In[68]:


# =========================================
# FINAL COMPARISON :
# ORIGINAL vs ALL MODELS
# =========================================

import matplotlib.pyplot as plt

# =========================================
# FIGURE SIZE
# =========================================

plt.figure(figsize=(20,10))

# =========================================
# ORIGINAL IMAGE
# =========================================

plt.subplot(2,3,1)

plt.imshow(test_image)

plt.title("Original Image")

plt.axis("off")

# =========================================
# AUTOENCODER
# =========================================

plt.subplot(2,3,2)

plt.imshow(reconstructed_image)

plt.title("Autoencoder")

plt.axis("off")

# =========================================
# DENOISING AUTOENCODER
# =========================================

plt.subplot(2,3,3)

plt.imshow(denoise_reconstructed)

plt.title("Denoising AE")

plt.axis("off")

# =========================================
# U-NET
# =========================================

plt.subplot(2,3,4)

plt.imshow(unet_reconstructed)

plt.title("U-Net")

plt.axis("off")

# =========================================
# SIMPLIFIED GAN
# =========================================

plt.subplot(2,3,5)

plt.imshow(gan_reconstructed)

plt.title("Simplified GAN")

plt.axis("off")

# =========================================
# PIX2PIX
# =========================================

plt.subplot(2,3,6)

plt.imshow(pix2pix_reconstructed)

plt.title("Pix2Pix")

plt.axis("off")

# =========================================
# DISPLAY
# =========================================

plt.tight_layout()

plt.show()

print("Final Model Comparison Generated Successfully")


# In[69]:


# =========================================
# SAVE FINAL COMPARISON FIGURE
# =========================================

comparison_save_path = r"C:\Users\money\OneDrive\Desktop\Final_Model_Comparison.jpg"

plt.savefig(
    comparison_save_path,
    dpi=300,
    bbox_inches='tight'
)

print("Final Comparison Figure Saved Successfully")

print("Saved Location :")

print(comparison_save_path)


# In[71]:


# =========================================
# FINAL CONFUSION MATRIX COMPARISON
# FIXED VERSION
# =========================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

# =========================================
# CREATE FIGURE
# =========================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18,12)
)

# =========================================
# AUTOENCODER
# =========================================

cm_auto = np.array([
    [11097, 696],
    [550, 4041]
])

disp1 = ConfusionMatrixDisplay(
    confusion_matrix=cm_auto
)

disp1.plot(
    ax=axes[0,0],
    cmap='Blues',
    colorbar=False
)

axes[0,0].set_title("Autoencoder")

# =========================================
# DENOISING AE
# =========================================

cm_dae = np.array([
    [11130, 663],
    [672, 3919]
])

disp2 = ConfusionMatrixDisplay(
    confusion_matrix=cm_dae
)

disp2.plot(
    ax=axes[0,1],
    cmap='Blues',
    colorbar=False
)

axes[0,1].set_title("Denoising AE")

# =========================================
# U-NET
# =========================================

cm_unet = np.array([
    [11365, 428],
    [169, 4422]
])

disp3 = ConfusionMatrixDisplay(
    confusion_matrix=cm_unet
)

disp3.plot(
    ax=axes[0,2],
    cmap='Blues',
    colorbar=False
)

axes[0,2].set_title("U-Net")

# =========================================
# SIMPLIFIED GAN
# =========================================

cm_gan = np.array([
    [11382, 411],
    [397, 4194]
])

disp4 = ConfusionMatrixDisplay(
    confusion_matrix=cm_gan
)

disp4.plot(
    ax=axes[1,0],
    cmap='Blues',
    colorbar=False
)

axes[1,0].set_title("Simplified GAN")

# =========================================
# PIX2PIX
# =========================================

cm_pix2pix = np.array([
    [11088, 705],
    [376, 4215]
])

disp5 = ConfusionMatrixDisplay(
    confusion_matrix=cm_pix2pix
)

disp5.plot(
    ax=axes[1,1],
    cmap='Blues',
    colorbar=False
)

axes[1,1].set_title("Pix2Pix")

# =========================================
# EMPTY LAST PLOT
# =========================================

axes[1,2].axis("off")

# =========================================
# MAIN TITLE
# =========================================

plt.suptitle(
    "Confusion Matrix Comparison of All Models",
    fontsize=18
)

# =========================================
# LAYOUT
# =========================================

plt.tight_layout()

plt.show()

print("All Model Confusion Matrix Comparison Generated Successfully")


# In[72]:


# =========================================
# SAVE FINAL CONFUSION MATRIX COMPARISON
# =========================================

save_path = r"C:\Users\money\OneDrive\Desktop\All_Model_Confusion_Matrix.jpg"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)

print("Final Confusion Matrix Comparison Saved Successfully")

print("Saved Location :")

print(save_path)


# In[75]:


# =========================================
# COMBINED BAR + LINE CHART
# PRIMARY AXIS  : PSNR + SSIM (BAR)
# SECONDARY AXIS: NMSE + IoU + Dice (LINE)
# =========================================

import matplotlib.pyplot as plt
import numpy as np

# =========================================
# MODEL NAMES
# =========================================

models = [
    'Autoencoder',
    'Denoising AE',
    'U-Net',
    'Simplified GAN',
    'Pix2Pix'
]

# =========================================
# METRIC VALUES
# =========================================

nmse_values = [
    0.000847,
    0.001066,
    0.000542,
    0.000860,
    0.000863
]

psnr_values = [
    23.8940,
    22.7079,
    31.3728,
    27.1349,
    24.0869
]

ssim_values = [
    0.7072,
    0.6007,
    0.9186,
    0.7926,
    0.6895
]

iou_values = [
    0.7643,
    0.7459,
    0.8811,
    0.8385,
    0.7959
]

dice_values = [
    0.8664,
    0.8545,
    0.9368,
    0.9121,
    0.8863
]

# =========================================
# X POSITION
# =========================================

x = np.arange(len(models))

width = 0.30

# =========================================
# CREATE FIGURE
# =========================================

fig, ax1 = plt.subplots(
    figsize=(16,8)
)

# =========================================
# PRIMARY AXIS : BAR CHART
# =========================================

bar1 = ax1.bar(
    x - width/2,
    psnr_values,
    width,
    label='PSNR'
)

bar2 = ax1.bar(
    x + width/2,
    ssim_values,
    width,
    label='SSIM'
)

# =========================================
# PRIMARY LABELS
# =========================================

ax1.set_ylabel(
    'PSNR / SSIM'
)

ax1.set_xlabel(
    'Models'
)

ax1.set_xticks(x)

ax1.set_xticklabels(
    models,
    rotation=10
)

# =========================================
# SHOW BAR VALUES
# =========================================

for bars in [bar1, bar2]:

    for bar in bars:

        height = bar.get_height()

        ax1.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

# =========================================
# SECONDARY AXIS : LINE CHART
# =========================================

ax2 = ax1.twinx()

line1 = ax2.plot(
    x,
    nmse_values,
    marker='o',
    linewidth=3,
    linestyle='--',
    label='NMSE'
)

line2 = ax2.plot(
    x,
    iou_values,
    marker='s',
    linewidth=3,
    linestyle='-.',
    label='IoU'
)

line3 = ax2.plot(
    x,
    dice_values,
    marker='d',
    linewidth=3,
    linestyle=':',
    label='Dice'
)

# =========================================
# SECONDARY LABEL
# =========================================

ax2.set_ylabel(
    'NMSE / IoU / Dice'
)

# =========================================
# SHOW LINE VALUES
# =========================================

for i, value in enumerate(nmse_values):

    ax2.text(
        x[i],
        value,
        f'{value:.4f}',
        fontsize=9
    )

for i, value in enumerate(iou_values):

    ax2.text(
        x[i],
        value,
        f'{value:.3f}',
        fontsize=9
    )

for i, value in enumerate(dice_values):

    ax2.text(
        x[i],
        value,
        f'{value:.3f}',
        fontsize=9
    )

# =========================================
# TITLE
# =========================================

plt.title(
    'Combined Performance Comparison of All Models',
    fontsize=18
)

# =========================================
# COMBINED LEGEND
# =========================================

handles1, labels1 = ax1.get_legend_handles_labels()

handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc='upper left'
)

# =========================================
# GRID
# =========================================

ax1.grid(True)

# =========================================
# LAYOUT
# =========================================

plt.tight_layout()

plt.show()

print("Combined Bar + Line Chart Generated Successfully")


# In[76]:


# =========================================
# SAVE COMBINED CHART
# =========================================

save_path = r"C:\Users\money\OneDrive\Desktop\Combined_Bar_Line_Chart.jpg"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches='tight'
)

print("Combined Chart Saved Successfully")

print("Saved Location :")

print(save_path)


# In[78]:


# =========================================
# IMPROVED PIX2PIX RESTORATION
# =========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# CONVERT PIX2PIX IMAGE
# =========================================

pix2pix_uint8 = (
    pix2pix_reconstructed * 255
).astype(np.uint8)

# =========================================
# ORIGINAL IMAGE
# =========================================

original_uint8 = (
    test_image * 255
).astype(np.uint8)

# =========================================
# STEP 1 : EDGE-PRESERVING FILTER
# =========================================

smooth_image = cv2.edgePreservingFilter(
    pix2pix_uint8,
    flags=1,
    sigma_s=60,
    sigma_r=0.4
)

# =========================================
# STEP 2 : BILATERAL FILTER
# =========================================

bilateral = cv2.bilateralFilter(
    smooth_image,
    d=9,
    sigmaColor=75,
    sigmaSpace=75
)

# =========================================
# STEP 3 : LIGHT SHARPENING
# =========================================

gaussian = cv2.GaussianBlur(
    bilateral,
    (0,0),
    3
)

restored = cv2.addWeighted(
    bilateral,
    1.5,
    gaussian,
    -0.5,
    0
)

# =========================================
# STEP 4 : CRACK REGION SMOOTH REPAIR
# =========================================

difference = cv2.absdiff(
    original_uint8,
    restored
)

gray_diff = cv2.cvtColor(
    difference,
    cv2.COLOR_RGB2GRAY
)

# THRESHOLD DIFFERENCE

_, mask = cv2.threshold(
    gray_diff,
    25,
    255,
    cv2.THRESH_BINARY
)

# INPAINTING

final_restored = cv2.inpaint(
    restored,
    mask,
    3,
    cv2.INPAINT_TELEA
)

# =========================================
# DISPLAY RESULTS
# =========================================

plt.figure(figsize=(18,6))

# ORIGINAL

plt.subplot(1,3,1)

plt.imshow(test_image)

plt.title("Original Image")

plt.axis("off")

# PIX2PIX

plt.subplot(1,3,2)

plt.imshow(pix2pix_reconstructed)

plt.title("Pix2Pix Reconstruction")

plt.axis("off")

# RESTORED

plt.subplot(1,3,3)

plt.imshow(final_restored)

plt.title("Improved Restored Image")

plt.axis("off")

plt.show()

print("Improved Restoration Generated Successfully")


# In[ ]:




