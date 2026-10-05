# 🛣️ Pothole Detection System

An AI-powered **Pothole Detection System** built using **YOLOv8 Instance Segmentation** and OpenCV. The system detects potholes in road images and videos, generates segmentation masks, calculates the damaged road area, and estimates the percentage of road damage.

## 🚀 Features

- 🔍 Pothole detection using **YOLOv8 Segmentation**
- 🎭 Pixel-level pothole segmentation using masks
- 🖼️ Image-based pothole detection
- 🎥 Video-based pothole detection
- 📊 Model validation and performance analysis
- 📈 Training and validation loss visualization
- 📐 Pothole area calculation in pixels
- 🛣️ Road damage percentage estimation
- 🎬 Road damage annotation on video frames
- 📦 Export trained model to **ONNX**
- ☁️ Google Colab + Google Drive support
- ⚡ GPU training support using NVIDIA T4

## 🧠 Technology Stack

- **Python**
- **YOLOv8 / Ultralytics**
- **OpenCV**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Pillow**
- **PyYAML**
- **Google Colab**
- **ONNX**

## 📂 Project Structure

```text
Pothole-Detection-System/
│
├── Pothole_detection.ipynb
├── README.md
│
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
└── runs/
    └── segment/
        └── train/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── results.png
            ├── results.csv
            ├── confusion_matrix.png
            ├── confusion_matrix_normalized.png
            ├── BoxPR_curve.png
            ├── MaskPR_curve.png
            └── ...
```

> **Note:** The dataset, trained weights, and generated videos are not required to be committed to GitHub if they are too large. The notebook expects the dataset to be available in Google Drive.

## 📓 Google Colab

You can open and run the notebook directly in Google Colab:

**[Open in Google Colab](https://colab.research.google.com/github/ayushdeval-github/Pothole-Detection-System/blob/main/Pothole_detection.ipynb)**

The notebook automatically installs the Ultralytics library and loads the YOLOv8 segmentation model.

## 📦 Installation

Install the required dependencies:

```bash
pip install ultralytics
```

The notebook uses the following major Python libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import yaml

from PIL import Image
from ultralytics import YOLO
```

## 📊 Dataset Setup

The project uses a custom YOLOv8 segmentation dataset stored in Google Drive.

The expected dataset path is:

```text
/content/drive/MyDrive/Pothole_Segmentation_YOLOv8
```

The dataset should contain a `data.yaml` configuration file along with training and validation images.

Expected structure:

```text
Pothole_Segmentation_YOLOv8/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── data.yaml
└── sample_video.mp4
```

## 🏋️ Model Training

The project starts with the pretrained YOLOv8 Nano segmentation model:

```python
model = YOLO("yolov8n-seg.pt")
```

The custom model is trained with:

- **Epochs:** 150
- **Image Size:** 640
- **Batch Size:** 16
- **Patience:** 15
- **Initial Learning Rate:** 0.0001
- **Dropout:** 0.25
- **GPU:** CUDA device 0
- **Random Seed:** 42

These settings are defined in the notebook's training pipeline.

Example:

```python
results = model.train(
    data=yaml_file_path,
    epochs=150,
    imgsz=640,
    patience=15,
    batch=16,
    optimizer="auto",
    lr0=0.0001,
    lrf=0.01,
    dropout=0.25,
    device=0,
    seed=42
)
```

## 📈 Model Evaluation

After training, the best model weights are loaded:

```python
best_model_path = "/content/runs/segment/train/weights/best.pt"

best_model = YOLO(best_model_path)
metrics = best_model.val(split="val")
```

The project evaluates the model using metrics related to:

- Bounding Box Precision
- Bounding Box Recall
- Bounding Box F1 Score
- Mask Precision
- Mask Recall
- Mask F1 Score
- Precision-Recall curves
- Confusion Matrix
- Training and validation losses

The notebook generates separate plots for bounding-box and segmentation losses and performance curves.

## 🖼️ Image Detection

The trained model can detect potholes in individual road images:

```python
results = best_model.predict(
    source=image_path,
    imgsz=640,
    conf=0.5
)
```

The system generates annotated images containing detected potholes and their segmentation masks.

## 🎥 Video Detection

The system can also process road videos and detect potholes frame-by-frame.

```python
best_model.predict(
    source=video_path,
    save=True,
    conf=0.1
)
```

The resulting video contains the detected potholes and segmentation results.

## 🛣️ Road Damage Assessment

One of the key features of the project is estimating the percentage of road surface affected by potholes.

For each detected pothole, the segmentation mask is converted into a binary mask and the contour area is calculated using OpenCV.

```python
area = cv2.contourArea(contour)
```

The total damaged area is then calculated:

```python
total_area += area
```

Finally, the estimated road damage percentage is:

```python
percentage_damage = (total_area / image_area) * 100
```

The system can therefore produce information such as:

```text
Area of Pothole 1: XXXX pixels
Area of Pothole 2: XXXX pixels
--------------------------------------------------
Total Damaged Area by Potholes: XXXX pixels
Total Pixels in Image: XXXXX pixels
Percentage of Road Damaged: XX.XX%
```

This calculation is implemented directly in the notebook.

## 🎬 Video-Based Damage Assessment

For video processing, the system calculates the damaged-area percentage for each frame and applies a moving average over the latest 10 frames to smooth the result.

The processed video is annotated with:

```text
Road Damage: XX.XX%
```

The system then saves the processed video for visualization.

## 📦 Model Export

The trained YOLOv8 model can be exported to ONNX:

```python
best_model.export(format="onnx")
```

This allows the trained model to be used in environments that support ONNX inference.

## ⚙️ Hardware

The notebook is configured to use a **GPU**, specifically an NVIDIA **T4 GPU** in Google Colab.

```text
Python 3
GPU: NVIDIA T4
Framework: Ultralytics YOLO
```

The notebook metadata confirms the T4 GPU configuration.

## 🔄 Workflow

```text
Road Dataset
     │
     ▼
Dataset Verification
     │
     ▼
YOLOv8 Segmentation Model
     │
     ▼
Custom Model Training
     │
     ▼
Model Validation
     │
     ├───────────────┐
     ▼               ▼
Image Detection   Video Detection
     │               │
     ▼               ▼
Pothole Masks    Frame-by-Frame Detection
     │               │
     └───────┬───────┘
             ▼
     Pothole Area Calculation
             │
             ▼
    Road Damage Percentage
             │
             ▼
       Damage Assessment
```

## ⚠️ Important Notes

1. The notebook expects the dataset to be available in Google Drive.
2. Make sure `data.yaml` exists inside the dataset directory.
3. GPU acceleration is recommended for training.
4. The calculated damaged area is expressed in **pixels**, not physical units such as square meters.
5. Road damage percentage is an image-based estimate based on the detected segmentation masks.
6. Large datasets, videos, and model weights should generally not be committed directly to GitHub.

## 🔮 Future Improvements

- Real-time pothole detection using a webcam
- Deployment as a Streamlit web application
- Mobile application integration
- GPS-based pothole location mapping
- Severity classification: Low / Medium / High
- Real-world area estimation in square meters
- Automatic road-condition reports
- Cloud-based inference API
- Integration with Google Maps
- Real-time dashboard for road authorities

## 👨‍💻 Author

**Ayush Deval**

B.Tech Computer Science & Engineering

GitHub: [@ayushdeval-github](https://github.com/ayushdeval-github)

---

⭐ If you find this project useful, consider giving the repository a star!
