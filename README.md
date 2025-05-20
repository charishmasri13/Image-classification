
# 🧠 Multi-Class Image Classification using CNN (Cat, Dog, Elephant, Lion)

This project demonstrates how to build a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images into four categories: **Cat**, **Dog**, **Elephant**, and **Lion**. The project includes both training and testing scripts written in Jupyter notebooks.

---

## 📁 Files Included

- `image_training.ipynb`: Training notebook for binary classification (initial version).
- `image_testing.ipynb`: Testing notebook for binary classification.
- `multiple_image_training.ipynb`: Updated notebook for training a multi-class classification model.
- `multiple_image_testing.ipynb`: Testing and prediction notebook for the multi-class model.

---

## 🗂️ Dataset Structure

Organize your training and testing folders like this:

```
/training/
├── Cat/
├── Dog/
├── Elephant/
└── Lion/

/testing/
├── Cat/
├── Dog/
├── Elephant/
└── Lion/
```

Each folder must contain images related to its class.

---

## 🔨 Model Architecture

- **Input Layer**: (64x64x3)
- **Conv2D**: 32 filters, 3x3 kernel
- **MaxPooling2D**
- **Flatten**
- **Dense**: 128 neurons with ReLU
- **Output Dense**: 4 neurons (softmax) for 4-class classification

---

## ⚙️ Training Settings

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Training:

```python
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)
```

---

## 🔍 Predicting a Single Image

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = 'path/to/image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

pred = model.predict(img_array)
class_labels = list(train_generator.class_indices.keys())
predicted_label = class_labels[np.argmax(pred[0])]

print("Predicted class:", predicted_label)
```

---

## 🔎 Predicting Multiple Images in a Folder

```python
import os
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Path to the folder containing test images
folder_path = '/content/test_images/'  # Change to your folder path

# Get class labels from training generator
class_labels = list(train_generator.class_indices.keys())

# Predict each image in the folder
for img_file in os.listdir(folder_path):
    if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(folder_path, img_file)
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction[0])]

        print(f"Image: {img_file} → Predicted Class: {predicted_class}")

        # Optional: Display image with predicted label
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class}")
        plt.axis('off')
        plt.show()
```

---

## 📊 Accuracy Example

You can track training progress with `model.fit()` and visualize accuracy and loss using `matplotlib` if desired.

---

## ✅ To-Do

- [ ] Add image augmentation
- [ ] Implement model saving and loading
- [ ] Deploy via Streamlit or Flask for real-time use
- [ ] Visualize predictions with matplotlib

---

## 📌 Requirements

Install required packages with:

```bash
pip install tensorflow numpy matplotlib
```
