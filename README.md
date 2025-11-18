## 🩺 Chest X-Ray Pneumonia Detector with Grad-CAM

This project provides a desktop application for classifying chest X-ray images to detect the presence of pneumonia. Built using **Tkinter** and **PyTorch**, the system not only classifies images but also incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the exact areas of the X-ray image that led to the model's decision, enhancing transparency and trust in the AI diagnosis.



### ✨ Features

* **Pneumonia Classification:** Classifies uploaded X-ray images into one of three categories:
    * **Normal** (Green)
    * **Pneumonia** (Red)
    * **Unknown** (Orange - for uncertain or out-of-distribution inputs)
* **Visual Explainability:** Integrates **Grad-CAM** to generate a heatmap overlay, highlighting the region(s) of interest (ROI) the model focused on for its prediction.
* **Confidence Scoring:** Displays the prediction confidence percentage for the classified class.
* **Desktop GUI:** User-friendly interface built with the `tkinter` library.



### 🚀 Setup and Installation

To run this application, you will need Python and several essential libraries, including PyTorch and OpenCV.

#### 1. Prerequisites

* Python 3.x
* The trained PyTorch model file: **`pneumonia_unknown_model.pth`**
* (Optional) A background image file: **`background.png`**

#### 2. Install Dependencies

Install the required libraries using pip:

```bash
# Core Libraries
pip install torch torchvision numpy
# GUI and Image Processing Libraries
pip install pillow opencv-python
