# 🌟 Chest X-Ray Pneumonia Detection Using ResNet18 + Explainable AI (Grad-CAM) + Tkinter GUI

An advanced Deep Learning-powered Desktop Application that detects Pneumonia, differentiates Normal X-rays, and even flags Unknown/Non-X-ray images using a custom-trained ResNet18 model.

This project also integrates Explainable AI via Grad-CAM, providing visual heatmaps that show where the model is focusing while making predictions.

✔ Fully Offline (no internet needed)  
✔ Clean Tkinter GUI  
✔ Medical-Grade Explainability  
✔ 3-Class Classification: NORMAL, PNEUMONIA, UNKNOWN

---

## 📑 Table of Contents

- [📌 Overview](#-overview)
- [📂 Dataset Description](#-dataset-description)
- [🚀 Features](#-features)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚙️ Installation Guide](#️-installation-guide)
- [🖼️ Application Preview](#️-application-preview)
- [🏃 Running the App](#-running-the-app)
- [📈 Model Results](#-model-results)
- [🧠 Explainable AI (Grad-CAM)](#-explainable-ai-grad-cam)
- [📦 Project Structure](#-project-structure)
- [🔮 Future Enhancements](#-future-enhancements)
- [🙌 Acknowledgements](#-acknowledgements)

---

## 📌 Overview

Pneumonia is a life-threatening lung infection that requires early and accurate diagnosis.
Radiologists analyze chest X-rays manually, which is time-consuming and error-prone.

This project builds an **AI-based Diagnostic Assistant** that:

- Automatically classifies chest X-ray images into:
  **NORMAL**, **PNEUMONIA**, **UNKNOWN**

- Displays **Grad-CAM heatmaps** for transparency and medical interpretability.

- Provides a user-friendly GUI built using **Tkinter**.

- Runs fully offline, without GPUs or heavy dependencies during inference.

This makes the system suitable for:

- Hospitals
- Mobile clinics
- Low-resource settings
- Academic research

---

## 📂 Dataset Description

The dataset used for training and evaluation is a balanced 3-class dataset:

🔗 **Kaggle Dataset Link**

👉 [https://www.kaggle.com/datasets/vklikith/pneumonia-balanced](https://www.kaggle.com/datasets/vklikith/pneumonia-balanced)

### Dataset Structure
```
pneumonia-balanced/
 └── Balanced/
      ├── train/
      │     ├── NORMAL
      │     ├── PNEUMONIA
      │     └── UNKNOWN
      ├── val/
      └── test/
```

### Class Distribution (Balanced)

- **NORMAL** – equal representation
- **PNEUMONIA** – equal representation
- **UNKNOWN** – includes non-X-ray images to improve robustness

---

## 🚀 Features

### 🩺 1. Pneumonia Detection

Classifies X-ray images using a custom-trained ResNet18 model.

### 🧪 2. Unknown Image Identification

If a user uploads a non-X-ray or irrelevant image, the model predicts **UNKNOWN**.

### 🔥 3. Explainability with Grad-CAM

Heatmaps show which areas the model used for its prediction → increases trust.

### 💻 4. Graphical User Interface

Intuitive Tkinter GUI:

- Upload X-ray images
- View predictions
- See Grad-CAM overlay

### 📊 5. Full EDA Included

- Class distribution
- Pixel intensity analysis
- Heatmaps
- Dimension scatter plots

### 📈 6. High Accuracy

Achieves excellent performance on train/val/test splits (Confusion Matrix + ROC curves included).

---

## 🧠 Model Architecture

- **Base Model:** ResNet18
- **Pretrained Weights:** ImageNet
- **Modified Output Layer:** 3 neurons → NORMAL, PNEUMONIA, UNKNOWN
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam

### Evaluation Metrics:

- Accuracy
- Confusion Matrix
- ROC-AUC
- Classification Report

---

## 📊 Exploratory Data Analysis (EDA)

Performed on Kaggle Notebook:

- Class distribution chart
- Train/Val/Test split distribution
- Heatmap of class counts
- Sample X-ray visualization
- Pixel intensity KDE
- Correlation heatmap
- Image dimension scatter

**EDA ensures:**

✔ Balanced dataset  
✔ Proper preprocessing  
✔ No corrupted images  
✔ Consistent dimensions

---

## 🛠️ Tech Stack

| Component       | Technology                      |
|-----------------|---------------------------------|
| Deep Learning   | PyTorch, TorchVision            |
| Model           | ResNet18 + Custom FC Layer      |
| Explainability  | Grad-CAM                        |
| GUI             | Tkinter (Python Standard Library)|
| Visualization   | Matplotlib, Seaborn             |
| Dataset         | Kaggle                          |
| Notebook        | Kaggle GPU Runtime              |

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Likith-2004/Chest-X-Ray-Pneumonia-Detection.git
cd Chest-X-Ray-Pneumonia-Detection
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv chest
source chest/bin/activate     # Mac/Linux
chest\Scripts\activate        # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Place Model File

Download from Kaggle training:

- `pneumonia_unknown_model.pth`

Place it inside the project folder.

---

## 🖼️ Application Preview

### 🟦 GUI Interface

- **Left:** Original Uploaded Image
- **Right:** Grad-CAM Heatmap
- **Bottom/Top:** Prediction with Confidence Score

Smooth, simple, and professional.

---

## 🏃 Running the App
```bash
python app.py
```

Then:

1. Click **Upload X-Ray**
2. See prediction instantly
3. View Grad-CAM heatmap
4. **UNKNOWN** prediction appears for irrelevant images

---

## 📈 Model Results

### Metrics Achieved:

- High Train & Validation Accuracy
- High Test Accuracy
- Strong class separation in Confusion Matrix
- High ROC-AUC scores for all classes

### Outputs:

✔ `confusion_matrix.png`  
✔ `roc_curves.png`  
✔ `classification_report.txt`

---

## 🧠 Explainable AI (Grad-CAM)

Grad-CAM was integrated to:

- Highlight infection regions
- Provide medical interpretability
- Build trust with healthcare professionals

### Heatmaps show:

- Hotspots in lungs for pneumonia
- Clear lungs for normal
- Random focus for unknown images

---

## 📦 Project Structure
```
📁 pneumonia-detection/
│── app.py                     → Tkinter GUI
│── pneumonia_unknown_model.pth → Trained Model
│── eda.ipynb                  → EDA Notebook
│── training.ipynb             → Training Notebook
│── requirements.txt
│── README.md
│── outputs/
     ├── confusion_matrix.png
     ├── roc_curves.png
     └── classification_report.txt
```

---

## 🔮 Future Enhancements

- Deploy as a Flask or Streamlit web app
- Add more diseases (Tuberculosis, COVID-19)
- Use EfficientNet or Vision Transformers
- Optimize model for mobile deployment
- Add batch prediction mode

---

## 🙌 Acknowledgements

Special thanks to:

- **Kaggle** for dataset hosting
- **PyTorch team** for open-source deep learning tools
- **Stanford & NIH** Chest X-Ray research teams
- All contributors & researchers working on medical AI
