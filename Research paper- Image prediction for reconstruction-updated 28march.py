#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

BASE_DIR = r"C:\Users\money\OneDrive\Desktop\Reconstruction dataset"

print("Train folders:", os.listdir(os.path.join(BASE_DIR, "train")))
print("Test folders:", os.listdir(os.path.join(BASE_DIR, "test")))


# In[2]:


def count_images(folder):
    for category in os.listdir(folder):
        path = os.path.join(folder, category)
        print(category, ":", len(os.listdir(path)))

print("\nTrain Data:")
count_images(os.path.join(BASE_DIR, "train"))

print("\nTest Data:")
count_images(os.path.join(BASE_DIR, "test"))


# In[3]:


#2.1: Import Required Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# In[4]:


#2.2: Define Parameters
IMG_SIZE = 224
BATCH_SIZE = 16

BASE_DIR = r"C:\Users\money\OneDrive\Desktop\Reconstruction dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")


# In[5]:


#2.3: Create Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,          # normalize pixel values
    rotation_range=20,       # rotate images
    zoom_range=0.2,          # zoom
    horizontal_flip=True,    # flip
    shear_range=0.2,         # distortion
    validation_split=0.2     # 80% train / 20% validation
)


# In[6]:


#2.4: Load Training Data
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)


# In[7]:


#2.6: Load Test Data
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# In[8]:


#VALIDATION
val_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)


# In[9]:


#2.7: Verify Output
print("Class Indices:", train_data.class_indices)
print("Train samples:", train_data.samples)
print("Validation samples:", val_data.samples)
print("Test samples:", test_data.samples)


# # STEP 3: BUILD & TRAIN FIRST MODEL (BASIC CNN

# In[10]:


#3.1: BUILD CNN MODEL
from tensorflow.keras import layers, models

def build_cnn_model():
    model = models.Sequential()

    # Layer 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(layers.MaxPooling2D(2,2))

    # Layer 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Flatten
    model.add(layers.Flatten())

    # Dense
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))  # 3 classes

    return model

cnn_model = build_cnn_model()
cnn_model.summary()


# In[11]:


# 3.2: COMPILE MODEL
cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[12]:


#3.3: TRAIN MODEL
history = cnn_model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


# In[13]:


#3.4: SAVE MODEL
cnn_model.save("cnn_model.h5")


# In[14]:


#3.5: CHECK ACCURACY GRAPH
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("CNN Accuracy")
plt.legend(['Train', 'Validation'])
plt.show()


# # STEP 4: MODEL EVALUATION (CONFUSION MATRIX + F1)

# In[15]:


#4.1: PREDICTION
import numpy as np

preds = cnn_model.predict(test_data)
y_pred = np.argmax(preds, axis=1)


# In[16]:


#4.2: CLASSIFICATION REPORT
from sklearn.metrics import classification_report

print(classification_report(test_data.classes, y_pred))


# In[17]:


#4.3: CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_data.classes, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - CNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# # STEP 5: BUILD DEEP CNN (IMPROVED MODEL)

# In[18]:


#5.1: BUILD DEEP CNN
def build_deep_cnn():
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    return model

deep_cnn = build_deep_cnn()
deep_cnn.summary()


# In[19]:


#5.2: COMPILE
deep_cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[20]:


#5.3: TRAIN
history_deep = deep_cnn.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)


# In[21]:


#5.4: SAVE MODEL
deep_cnn.save("deep_cnn_model.h5")


# In[22]:


#5.5: ACCURACY GRAPH
plt.plot(history_deep.history['accuracy'])
plt.plot(history_deep.history['val_accuracy'])
plt.title("Deep CNN Accuracy")
plt.legend(['Train', 'Validation'])
plt.show()


# # COMPLETE STEP 5 (WITH EVALUATION)MODEL EVALUATION (CONFUSION MATRIX + F1)

# In[23]:


#5.6: PREDICT ON TEST DATA
import numpy as np

preds_deep = deep_cnn.predict(test_data)
y_pred_deep = np.argmax(preds_deep, axis=1)


# In[24]:


#5.7: CLASSIFICATION REPORT
from sklearn.metrics import classification_report

print("Deep CNN Classification Report:\n")
print(classification_report(test_data.classes, y_pred_deep))


# In[25]:


#5.8: CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_deep = confusion_matrix(test_data.classes, y_pred_deep)

plt.figure(figsize=(6,4))
sns.heatmap(cm_deep, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Deep CNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# # STEP 6: CNN + BATCH NORMALIZATION

# In[37]:


#6.1: BUILD MODEL
from tensorflow.keras import layers, models

def build_cnn_bn():
    model = models.Sequential()

    # 🔹 Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    # 🔹 Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    # 🔹 Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    # 🔹 Dense Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    return model

cnn_bn = build_cnn_bn()
cnn_bn.summary()


# In[38]:


#6.2: COMPILE MODEL
cnn_bn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[39]:


#6.3: TRAIN MODEL
history_bn = cnn_bn.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)


# In[40]:


#6.4: SAVE MODEL (IMPORTANT)
cnn_bn.save("cnn_bn_model.h5")


# In[41]:


#6.5: ACCURACY GRAPH
import matplotlib.pyplot as plt

plt.plot(history_bn.history['accuracy'])
plt.plot(history_bn.history['val_accuracy'])
plt.title("CNN + BatchNorm Accuracy")
plt.legend(['Train', 'Validation'])
plt.show()


# In[42]:


#6.6: EVALUATION (MANDATORY)
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

preds_bn = cnn_bn.predict(test_data)
y_pred_bn = np.argmax(preds_bn, axis=1)

print("Classification Report:\n")
print(classification_report(test_data.classes, y_pred_bn))


# In[43]:


#6.8: STORE METRICS (FOR FINAL COMPARISON)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_bn = accuracy_score(test_data.classes, y_pred_bn)
prec_bn = precision_score(test_data.classes, y_pred_bn, average='weighted')
rec_bn = recall_score(test_data.classes, y_pred_bn, average='weighted')
f1_bn = f1_score(test_data.classes, y_pred_bn, average='weighted')

print("Accuracy:", acc_bn)
print("Precision:", prec_bn)
print("Recall:", rec_bn)
print("F1 Score:", f1_bn)


# # 7: CNN + DROPOUT MODEL

# In[44]:


#7.1: BUILD MODEL
def build_cnn_dropout():
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(layers.MaxPooling2D(2,2))

    # Block 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Block 3
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # 🔥 key layer
    model.add(layers.Dense(3, activation='softmax'))

    return model

cnn_dropout = build_cnn_dropout()
cnn_dropout.summary()


# In[45]:


#7.2: COMPILE
cnn_dropout.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[46]:


#7.3: TRAIN
history_dropout = cnn_dropout.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)


# In[47]:


#7.4: EVALUATE
preds_dropout = cnn_dropout.predict(test_data)
y_pred_dropout = np.argmax(preds_dropout, axis=1)

from sklearn.metrics import classification_report
print(classification_report(test_data.classes, y_pred_dropout))


# In[48]:


#7.5: STORE METRICS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_dropout = accuracy_score(test_data.classes, y_pred_dropout)
prec_dropout = precision_score(test_data.classes, y_pred_dropout, average='weighted')
rec_dropout = recall_score(test_data.classes, y_pred_dropout, average='weighted')
f1_dropout = f1_score(test_data.classes, y_pred_dropout, average='weighted')

print(acc_dropout, prec_dropout, rec_dropout, f1_dropout)


# In[49]:


#7.6: ACCURACY GRAPH (TRAIN vs VALIDATION)
import matplotlib.pyplot as plt

# Accuracy
plt.figure()
plt.plot(history_dropout.history['accuracy'])
plt.plot(history_dropout.history['val_accuracy'])
plt.title("CNN + Dropout Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train', 'Validation'])
plt.show()


# In[50]:


# Loss
plt.figure()
plt.plot(history_dropout.history['loss'])
plt.plot(history_dropout.history['val_loss'])
plt.title("CNN + Dropout Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train', 'Validation'])
plt.show()


# # STEP 8: MINI RESNET

# In[51]:


#8.1: RESIDUAL BLOCK
from tensorflow.keras import layers

def residual_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, (3,3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, (3,3), padding='same')(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


# In[52]:


#8.2: BUILD MINI RESNET
from tensorflow.keras import models

def build_resnet():
    inputs = layers.Input(shape=(224,224,3))

    # Initial layer
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inputs)

    # Residual Block 1
    x = residual_block(x, 32)
    x = layers.MaxPooling2D()(x)

    # Residual Block 2
    x = residual_block(x, 32)
    x = layers.MaxPooling2D()(x)

    # Residual Block 3
    x = residual_block(x, 32)
    x = layers.MaxPooling2D()(x)

    # Dense
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    outputs = layers.Dense(3, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    return model

resnet_model = build_resnet()
resnet_model.summary()


# In[53]:


#8.3: COMPILE
resnet_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[54]:


#8.4: TRAIN
history_resnet = resnet_model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)


# In[55]:


#8.5: SAVE MODEL
resnet_model.save("resnet_model.h5")


# In[56]:


#8.6: ACCURACY GRAPH
import matplotlib.pyplot as plt

plt.plot(history_resnet.history['accuracy'])
plt.plot(history_resnet.history['val_accuracy'])
plt.title("ResNet Accuracy")
plt.legend(['Train','Validation'])
plt.show()


# In[57]:


#8.7: LOSS GRAPH
plt.plot(history_resnet.history['loss'])
plt.plot(history_resnet.history['val_loss'])
plt.title("ResNet Loss")
plt.legend(['Train','Validation'])
plt.show()


# In[58]:


#8.8: EVALUATION
import numpy as np
from sklearn.metrics import classification_report

preds_resnet = resnet_model.predict(test_data)
y_pred_resnet = np.argmax(preds_resnet, axis=1)

print(classification_report(test_data.classes, y_pred_resnet))


# In[59]:


#8.9: STORE METRICS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc_resnet = accuracy_score(test_data.classes, y_pred_resnet)
prec_resnet = precision_score(test_data.classes, y_pred_resnet, average='weighted')
rec_resnet = recall_score(test_data.classes, y_pred_resnet, average='weighted')
f1_resnet = f1_score(test_data.classes, y_pred_resnet, average='weighted')

print(acc_resnet, prec_resnet, rec_resnet, f1_resnet)


# # STEP 9: FINAL MODEL COMPARISON GRAPH

# In[62]:


models = ['CNN', 'Deep CNN', 'CNN+BN', 'CNN+Dropout', 'ResNet']

accuracy = [
    0.61,   # CNN (update if needed)
    0.44,   # Deep CNN
    0.38,   # BN
    0.0,    # Fill your dropout result
    0.508   # ResNet
]

f1_scores = [
    0.58,
    0.38,
    0.21,
    0.0,    # Fill
    0.447
]


# In[63]:


#PLOT GRAPH
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(models))

plt.figure(figsize=(10,5))

plt.bar(x-0.2, accuracy, 0.4, label='Accuracy')
plt.bar(x+0.2, f1_scores, 0.4, label='F1 Score')

plt.xticks(x, models)
plt.title("Model Comparison")
plt.legend()
plt.show()


# # STEP 10 (UPDATED): PREDICT IMAGE USING BEST MODEL

# In[64]:


#10.1: LOAD MODEL
from tensorflow.keras.models import load_model

model = load_model("cnn_model.h5")   # or "resnet_model.h5"


# In[66]:


#10.2: IMAGE PREDICTION FUNCTION
import cv2
import numpy as np
import matplotlib.pyplot as plt

def predict_image(img_path, model):

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    img_resized = cv2.resize(img, (224,224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Prediction
    pred = model.predict(img_input)
    pred_class = np.argmax(pred)

    class_names = ['major crack', 'minor crack', 'spalling']

    # Display
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {class_names[pred_class]}")
    plt.axis("off")
    plt.show()

    print("Prediction Probabilities:", pred)


# In[70]:


#10.3: RUN FOR TEST IMAGE
def predict_image(img_path, model):

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    img = cv2.imread(img_path)

    if img is None:
        print("❌ Error: Image not found. Check path.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    img_resized = cv2.resize(img, (224,224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Prediction
    pred = model.predict(img_input)
    pred_class = np.argmax(pred)

    class_names = ['major crack', 'minor crack', 'spalling']

    # Display
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {class_names[pred_class]}")
    plt.axis("off")
    plt.show()

    print("Prediction Probabilities:", pred)


# In[71]:


import random

category = random.choice(os.listdir(TRAIN_DIR))
img_name = random.choice(os.listdir(os.path.join(TRAIN_DIR, category)))

img_path = os.path.join(TRAIN_DIR, category, img_name)

predict_image(img_path, model)


# # STEP 3 (UPDATED): RANDOM IMAGE + ALL MODEL PREDICTION

# In[72]:


#3.1: RANDOM IMAGE SELECT
import os
import random

def get_random_image():
    
    categories = os.listdir(TRAIN_DIR)
    
    # Random category
    category = random.choice(categories)
    
    # Random image from that category
    img_name = random.choice(os.listdir(os.path.join(TRAIN_DIR, category)))
    
    img_path = os.path.join(TRAIN_DIR, category, img_name)
    
    return img_path, category


# In[73]:


#STEP 3.2: COMBINE WITH MODEL COMPARISON
def compare_models_random():

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    img_path, actual_class = get_random_image()

    img = cv2.imread(img_path)

    if img is None:
        print("Image not found")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    img_resized = cv2.resize(img, (224,224))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    class_names = ['major crack', 'minor crack', 'spalling']

    models = {
        "CNN": cnn_model,
        "Deep CNN": deep_cnn,
        "CNN+BN": cnn_bn,
        "Dropout": cnn_dropout,
        "ResNet": resnet_model
    }

    # Show image
    plt.imshow(img_rgb)
    plt.title(f"Actual: {actual_class}")
    plt.axis("off")
    plt.show()

    print("\n--- Model Predictions ---\n")

    for name, model in models.items():
        pred = model.predict(img_input)
        pred_class = np.argmax(pred)
        print(f"{name} → {class_names[pred_class]}")


# In[74]:


compare_models_random()


# # STEP 11: SEGMENTATION (DAMAGE AREA DETECTION)

# In[75]:


#random image
img_path, actual_class = get_random_image()


# In[76]:


#11.2: THRESHOLD SEGMENTATION (BASELINE)
import cv2
import matplotlib.pyplot as plt

def segment_damage(img_path):

    img = cv2.imread(img_path)

    if img is None:
        print("Image not found")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold (you can tune value later)
    _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    return img_rgb, mask


# In[77]:


#11.3: SHOW SEGMENTATION OUTPUT
img_path, actual_class = get_random_image()

original, mask = segment_damage(img_path)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(original)
plt.title(f"Original ({actual_class})")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Damage Mask")
plt.axis("off")

plt.show()


# In[78]:


#STEP 11.4: CALCULATE DAMAGE %
import numpy as np

def calculate_damage(mask):
    damage_pixels = np.sum(mask > 0)
    total_pixels = mask.size

    percent = (damage_pixels / total_pixels) * 100
    return percent

damage_percent = calculate_damage(mask)

print("Damage Percentage:", round(damage_percent,2), "%")


# # STEP 12: ADVANCED SEGMENTATION (U-NET MODEL)

# In[80]:


#12.1: CREATE TRAINING DATA (IMAGE + MASK)PREPARE DATASET
import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_segmentation_data(base_dir):

    X = []
    Y = []

    categories = os.listdir(base_dir)

    for category in categories:
        path = os.path.join(base_dir, category)

        for img_name in os.listdir(path):

            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Create mask using threshold
            _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

            X.append(img / 255.0)
            Y.append(mask / 255.0)

    X = np.array(X)
    Y = np.array(Y)

    Y = np.expand_dims(Y, axis=-1)

    return X, Y

X_train_seg, Y_train_seg = load_segmentation_data(TRAIN_DIR)

print(X_train_seg.shape, Y_train_seg.shape)


# In[81]:


#12.2: BUILD U-NET MODEL
from tensorflow.keras import layers, models

def build_unet():

    inputs = layers.Input((128,128,3))

    # Encoder
    c1 = layers.Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)

    # Bottleneck
    b = layers.Conv2D(128,(3,3),activation='relu',padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D()(b)
    c3 = layers.Conv2D(64,(3,3),activation='relu',padding='same')(u1)

    u2 = layers.UpSampling2D()(c3)
    c4 = layers.Conv2D(32,(3,3),activation='relu',padding='same')(u2)

    outputs = layers.Conv2D(1,(1,1),activation='sigmoid')(c4)

    model = models.Model(inputs, outputs)

    return model

unet_model = build_unet()
unet_model.summary()


# In[82]:


#12.3: COMPILE
unet_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# In[83]:


#12.4: TRAIN
history_unet = unet_model.fit(
    X_train_seg, Y_train_seg,
    epochs=10,
    batch_size=8
)


# In[84]:


#12.6.1: ACCURACY & LOSS GRAPH
import matplotlib.pyplot as plt

# Accuracy
plt.plot(history_unet.history['accuracy'])
plt.title("U-Net Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Loss
plt.plot(history_unet.history['loss'])
plt.title("U-Net Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


# In[88]:


#12.6.2: IoU METRIC (VERY IMPORTANT)
import numpy as np

def calculate_iou(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.uint8)

    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)

    iou = np.sum(intersection) / np.sum(union)
    return iou


# In[89]:


#12.6.3: DICE SCORE
def calculate_dice(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.uint8)

    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice


# In[90]:


#12.6.4: EVALUATE ON SAMPLE
idx = np.random.randint(0, len(X_train_seg))

y_true = Y_train_seg[idx]
y_pred = unet_model.predict(X_train_seg[idx:idx+1])[0]

iou = calculate_iou(y_true, y_pred)
dice = calculate_dice(y_true, y_pred)

print("IoU:", iou)
print("Dice Score:", dice)


# In[93]:


#12.5.1: TEST U-NET ON RANDOM IMAGE
img_path, actual_class = get_random_image()

predict_unet(img_path)


# In[94]:


#12.5.2: IMPROVED VISUAL (BEST FOR REPORT)
def predict_unet_visual(img_path):

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (128,128))

    img_input = np.expand_dims(img_resized/255.0, axis=0)

    pred = unet_model.predict(img_input)[0]
    pred_mask = (pred > 0.5).astype(np.uint8)

    # Convert for display
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12,4))

    # Original
    plt.subplot(1,3,1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Raw prediction
    plt.subplot(1,3,2)
    plt.imshow(pred.squeeze(), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

    # Binary mask
    plt.subplot(1,3,3)
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.title("Binary Damage Area")
    plt.axis("off")

    plt.show()


# In[95]:


predict_unet_visual(img_path)


# # STEP 13: DAMAGE AREA → MATERIAL CALCULATION

# In[115]:


#SELECT RANDOM IMAGE
img_path, actual_class = get_random_image()
print("Selected Class:", actual_class)


# In[116]:


#READ IMAGE
import cv2

img = cv2.imread(img_path)
img_resized = cv2.resize(img, (128,128))


# In[117]:


#U-NET PREDICTION (MASK GENERATION)
import numpy as np

img_input = np.expand_dims(img_resized / 255.0, axis=0)

pred = unet_model.predict(img_input)[0]

# Convert to binary mask
pred_mask = (pred > 0.5).astype(np.uint8)


# In[118]:


#COUNT PIXELS
damage_pixels = np.sum(pred_mask > 0)
total_pixels = pred_mask.size

print("Damage Pixels:", damage_pixels)
print("Total Pixels:", total_pixels)


# In[119]:


#CALCULATE DAMAGE %
damage_percent = (damage_pixels / total_pixels) * 100

print("Damage %:", round(damage_percent,2))


# In[120]:


#CONVERT TO REAL AREA
real_area = 100  # sq.ft

damaged_area = (damage_percent / 100) * real_area

print("Damaged Area:", round(damaged_area,2), "sq.ft")


# In[121]:


#CEMENT CALCULATION 1 sq.ft → 0.02 bags cement
cement = damaged_area * 0.02

print("Cement Required:", round(cement,2), "bags")


# In[122]:


#SAND CALCULATION 1 sq.ft → 0.005 m³ sand
sand = damaged_area * 0.005

print("Sand Required:", round(sand,3), "m³")


# In[123]:


#SHOW IMAGE + MASK
import matplotlib.pyplot as plt

img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(pred_mask.squeeze(), cmap='gray')
plt.title("Damage Mask")
plt.axis("off")

plt.show()


# # STEP 14: MANPOWER ESTIMATION (LABOR CALCULATION)

# In[128]:


#STEP 14: MANPOWER ESTIMATION (LABOR CALCULATION)Area = 40 sq.ft Workers2
def manpower_estimation(area):

    work_per_person_per_day = 20  # sq.ft

    workers = area / work_per_person_per_day

    return workers


# In[129]:


workers = manpower_estimation(damaged_area)

print("Workers Required:", round(workers,2))


# In[130]:


#14.2: TIME ESTIMATION
def time_estimation(area):

    work_per_day = 20

    days = area / work_per_day

    return days


# In[131]:


days = time_estimation(damaged_area)

print("Estimated Time:", round(days,2), "days")


# In[132]:


#14.3: FINAL COMBINED OUTPUT
print("\n----- FINAL PROJECT OUTPUT -----")

print("Damage %:", round(damage_percent,2), "%")
print("Damaged Area:", round(damaged_area,2), "sq.ft")

print("\nMaterial Requirement:")
print("Cement:", round(cement,2), "bags")
print("Sand:", round(sand,3), "m³")

print("\nLabor Requirement:")
print("Workers Needed:", round(workers,2))
print("Time Required:", round(days,2), "days")


# # 15: IMAGE RECONSTRUCTION 

# In[133]:


# PREPARE DATA
import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_reconstruction_data(base_dir):

    X = []

    categories = os.listdir(base_dir)

    for category in categories:
        path = os.path.join(base_dir, category)

        for img_name in os.listdir(path):

            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img / 255.0)

    X = np.array(X)

    return X

X_train_rec = load_reconstruction_data(TRAIN_DIR)

print("Dataset shape:", X_train_rec.shape)


# In[134]:


#15.2: BUILD AUTOENCODER
from tensorflow.keras import layers, models

def build_autoencoder():

    input_img = layers.Input(shape=(128,128,3))

    # Encoder
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)

    output = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    model = models.Model(input_img, output)

    return model

autoencoder = build_autoencoder()
autoencoder.summary()


# In[135]:


#15.3: COMPILE
autoencoder.compile(
    optimizer='adam',
    loss='mse'
)


# In[137]:


#15.6: TRAIN WITH VALIDATION SPLIT
history_ae = autoencoder.fit(
    X_train_rec, X_train_rec,
    epochs=10,
    batch_size=8,
    validation_split=0.2
)


# In[138]:


#15.8: PSNR (IMAGE QUALITY METRIC)
import numpy as np

def calculate_psnr(original, reconstructed):

    mse = np.mean((original - reconstructed) ** 2)

    if mse == 0:
        return 100

    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


# In[139]:


#15.9: SSIM (STRUCTURAL SIMILARITY)
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(original, reconstructed):

    original_gray = cv2.cvtColor((original*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    recon_gray = cv2.cvtColor((reconstructed*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    score = ssim(original_gray, recon_gray)

    return score


# In[140]:


#15.10: EVALUATE ON RANDOM IMAGE
idx = np.random.randint(0, len(X_train_rec))

original = X_train_rec[idx]
input_img = np.expand_dims(original, axis=0)

reconstructed = autoencoder.predict(input_img)[0]

psnr_val = calculate_psnr(original, reconstructed)
ssim_val = calculate_ssim(original, reconstructed)

print("PSNR:", round(psnr_val,2))
print("SSIM:", round(ssim_val,3))


# In[142]:


#15.11: FINAL OUTPUT (ORIGINAL vs RECONSTRUCTED)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(original)
plt.title("Original (Damaged)")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(reconstructed)
plt.title("Reconstructed")
plt.axis("off")

plt.show()


# In[144]:


#15.1: LOAD DATASET
import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_data(base_dir):
    X = []

    for category in os.listdir(base_dir):
        path = os.path.join(base_dir, category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img / 255.0)

    return np.array(X)

X_train_rec = load_data(TRAIN_DIR)

print("Dataset shape:", X_train_rec.shape)


# In[145]:


#15.2: ADD NOISE
noise_factor = 0.2

X_noisy = X_train_rec + noise_factor * np.random.normal(
    loc=0.0, scale=1.0, size=X_train_rec.shape
)

X_noisy = np.clip(X_noisy, 0., 1.)


# In[146]:


#15.3: BUILD DENOISING AUTOENCODER
from tensorflow.keras import layers, models

def build_autoencoder():

    input_img = layers.Input(shape=(128,128,3))

    # Encoder
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)

    # Decoder
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)

    output = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

    model = models.Model(input_img, output)

    return model

autoencoder = build_autoencoder()
autoencoder.summary()


# In[147]:


#15.4: COMPILE MODEL
autoencoder.compile(
    optimizer='adam',
    loss='mse'
)


# In[148]:


#15.5: TRAIN MODEL
history_ae = autoencoder.fit(
    X_noisy, X_train_rec,   # 🔥 noisy → clean
    epochs=20,
    batch_size=8,
    validation_split=0.2
)


# In[149]:


#15.6: LOSS GRAPH
import matplotlib.pyplot as plt

plt.plot(history_ae.history['loss'])
plt.plot(history_ae.history['val_loss'])

plt.title("Denoising Autoencoder Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['Train','Validation'])

plt.show()


# In[150]:


#15.7: PSNR + SSIM
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(original, reconstructed):
    original_gray = cv2.cvtColor((original*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    recon_gray = cv2.cvtColor((reconstructed*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return ssim(original_gray, recon_gray)


# In[151]:


#15.8: FINAL OUTPUT (ORIGINAL vs RECONSTRUCTED)
idx = np.random.randint(0, len(X_train_rec))

original = X_train_rec[idx]
noisy = X_noisy[idx]

input_img = np.expand_dims(noisy, axis=0)

reconstructed = autoencoder.predict(input_img)[0]

psnr_val = calculate_psnr(original, reconstructed)
ssim_val = calculate_ssim(original, reconstructed)

print("PSNR:", round(psnr_val,2))
print("SSIM:", round(ssim_val,3))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(noisy)
plt.title("Noisy Input")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(reconstructed)
plt.title("Reconstructed Image")
plt.axis("off")

plt.show()


# # U-NET FOR RECONSTRUCTION

# In[7]:


import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_data(base_dir):
    X = []

    for category in os.listdir(base_dir):
        path = os.path.join(base_dir, category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img / 255.0)

    return np.array(X)

# 🔥 SET YOUR PATH
TRAIN_DIR = r"C:\Users\money\OneDrive\Desktop\Reconstruction dataset\train"

X_train_rec = load_data(TRAIN_DIR)

print("Dataset loaded:", X_train_rec.shape)


# In[8]:


print(type(X_train_rec))
print(X_train_rec.shape)


# In[9]:


# Create Noisy Data
noise_factor = 0.2

X_noisy = X_train_rec + noise_factor * np.random.normal(
    loc=0.0, scale=1.0, size=X_train_rec.shape
)

X_noisy = np.clip(X_noisy, 0., 1.)

print("Noisy data created:", X_noisy.shape)


# In[10]:


#Train Model
history_unet = unet_recon.fit(
    X_noisy, X_train_rec,
    epochs=15,
    batch_size=8,
    validation_split=0.2
)


# In[11]:


idx = np.random.randint(0, len(X_noisy))

input_img = X_noisy[idx:idx+1]   # noisy input
original_img = X_train_rec[idx]  # ground truth


# In[12]:


idx = np.random.randint(0, len(X_noisy))

input_img = X_noisy[idx:idx+1]   # noisy input
original_img = X_train_rec[idx]  # ground truth


# In[ ]:


for i in range(3):

    idx = np.random.randint(0, len(X_noisy))

    input_img = X_noisy[idx:idx+1]
    original_img = X_train_rec[idx]

    recon_unet = unet_recon.predict(input_img)[0]

    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.imshow(original_img)
    plt.title("Original")

    plt.subplot(1,3,2)
    plt.imshow(input_img[0])
    plt.title("Noisy")

    plt.subplot(1,3,3)
    plt.imshow(recon_unet)
    plt.title("U-Net")

    plt.show()


# In[14]:


print("Train:", X_train_rec.shape)
print("Noisy:", X_noisy.shape)


# In[15]:


idx = 0

input_img = X_noisy[idx:idx+1]
original_img = X_train_rec[idx]


# In[16]:


import matplotlib.pyplot as plt

plt.imshow(input_img[0])
plt.title("Noisy Input")
plt.show()


# In[17]:


recon_unet = unet_recon.predict(input_img)
print(recon_unet.shape)


# In[18]:


recon_img = recon_unet[0]

plt.imshow(recon_img)
plt.title("U-Net Output")
plt.show()


# In[19]:


recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())

plt.imshow(recon_img)
plt.title("U-Net Normalized Output")
plt.show()


# In[20]:


#SHOW ALL TOGETHER
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(original_img)
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(input_img[0])
plt.title("Noisy")

plt.subplot(1,3,3)
plt.imshow(recon_img)
plt.title("U-Net Output")

plt.show()


# In[21]:


print(recon_img.min(), recon_img.max())


# In[22]:


#TRAIN MORE
history_unet = unet_recon.fit(
    X_noisy, X_train_rec,
    epochs=30,   # increase
    batch_size=8,
    validation_split=0.2
)


# In[23]:


#REDUCE NOISE
noise_factor = 0.1   # reduce from 0.2


# In[24]:


X_noisy = X_train_rec + noise_factor * np.random.normal(
    loc=0.0, scale=1.0, size=X_train_rec.shape
)
X_noisy = np.clip(X_noisy, 0., 1.)


# In[25]:


#IMPROVE U-NET (STRONGER MODEL)
from tensorflow.keras import layers, models

def build_unet_recon():

    inputs = layers.Input((128,128,3))

    # Encoder
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    # Bottleneck
    b = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)

    # Decoder
    u1 = layers.UpSampling2D((2,2))(b)
    u1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D((2,2))(c3)
    u2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(c4)

    return models.Model(inputs, outputs)

unet_recon = build_unet_recon()
unet_recon.compile(optimizer='adam', loss='mse')


# In[26]:


#TRAIN AGAIN
history_unet = unet_recon.fit(
    X_noisy, X_train_rec,
    epochs=30,
    batch_size=8,
    validation_split=0.2
)


# In[27]:


#SHOW OUTPUT
idx = 0

input_img = X_noisy[idx:idx+1]
original_img = X_train_rec[idx]

recon = unet_recon.predict(input_img)[0]

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(original_img)
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(input_img[0])
plt.title("Noisy")

plt.subplot(1,3,3)
plt.imshow(recon)
plt.title("U-Net Improved")

plt.show()


# # 17: GAN (IMPROVED & STABLE)

# In[37]:


print(X_train_rec.shape)
print(X_noisy.shape)


# In[45]:


from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


# In[46]:


def build_generator():

    inputs = layers.Input(shape=(128,128,3))

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)

    outputs = layers.Conv2D(3, (3,3), padding='same', activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')  # ✅ KEY CHANGE

    return model

generator = build_generator()


# In[47]:


history_gan = generator.fit(
    X_noisy, X_train_rec,
    epochs=20,
    batch_size=8,
    validation_split=0.2
)


# In[48]:


#Test output
idx = 0

test_img = X_noisy[idx:idx+1]
recon_gan = generator.predict(test_img)[0]


# In[49]:


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(X_train_rec[idx])
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(X_noisy[idx])
plt.title("Noisy")

plt.subplot(1,3,3)
plt.imshow(recon_gan)
plt.title("GAN (Simplified)")

plt.show()


# # 18: PIX2PIX 

# In[50]:


from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[51]:


#Pix2pix model
def build_pix2pix():

    inputs = layers.Input((128,128,3))

    # 🔽 Encoder
    e1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(e1)

    e2 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(e2)

    # 🔥 Bottleneck
    b = layers.Conv2D(256, (3,3), activation='relu', padding='same')(p2)

    # 🔼 Decoder
    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, e2])
    d1 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(d1)
    u2 = layers.concatenate([u2, e1])
    d2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(d2)

    return models.Model(inputs, outputs)

pix2pix = build_pix2pix()


# In[52]:


#Compile
pix2pix.compile(
    optimizer='adam',
    loss='mse'
)


# In[53]:


#Train
history_pix = pix2pix.fit(
    X_noisy, X_train_rec,
    epochs=25,
    batch_size=8,
    validation_split=0.2
)


# In[54]:


idx = 0

test_img = X_noisy[idx:idx+1]
recon_pix = pix2pix.predict(test_img)[0]


# In[55]:


plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(X_train_rec[idx])
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(X_noisy[idx])
plt.title("Noisy")

plt.subplot(1,3,3)
plt.imshow(recon_pix)
plt.title("Pix2Pix Output")

plt.show()


# # SHOW ALL RECONSTRUCTION IMAGE

# In[56]:


idx = 0

original = X_train_rec[idx]
noisy = X_noisy[idx]

# Already predicted (make sure these exist)
# reconstructed (Autoencoder)
# recon_unet
# recon_gan
# recon_pix


# In[64]:


idx = 0

original = X_train_rec[idx]
noisy = X_noisy[idx]

recon_unet = unet_recon.predict(X_noisy[idx:idx+1])[0]
recon_gan = generator.predict(X_noisy[idx:idx+1])[0]
recon_pix = pix2pix.predict(X_noisy[idx:idx+1])[0]


# In[66]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))

plt.subplot(2,3,1)
plt.imshow(original)
plt.title("Original")

plt.subplot(2,3,2)
plt.imshow(noisy)
plt.title("Noisy")


plt.subplot(2,3,4)
plt.imshow(recon_unet)
plt.title("U-Net")

plt.subplot(2,3,5)
plt.imshow(recon_gan)
plt.title("GAN")

plt.subplot(2,3,6)
plt.imshow(recon_pix)
plt.title("Pix2Pix")

plt.show()


# In[68]:


#Final Comparison MSE
def mse(a, b):
    return np.mean((a - b)**2)

results = {}

results['U-Net'] = mse(original, recon_unet)
results['GAN'] = mse(original, recon_gan)
results['Pix2Pix'] = mse(original, recon_pix)

for k,v in results.items():
    print(k, ":", round(v,4))


# In[69]:


import matplotlib.pyplot as plt

names = list(results.keys())
values = list(results.values())

plt.bar(names, values)
plt.title("Model Comparison (MSE)")
plt.ylabel("Error")
plt.show()


# In[ ]:




