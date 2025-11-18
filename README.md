### Chest X-Ray Pneumonia Detector with Grad-CAM

## Overview
The **Chest X-Ray Pneumonia Detector** project leverages a Convolutional Neural Network (CNN) model (ResNet-18) to classify chest X-ray images for the detection of pneumonia. By automating this detection process, the system enables fast and reliable pre-screening. The project is designed to enhance diagnostic confidence by including Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize the areas of the X-ray image that led to the model's prediction.


## Features
- **Pneumonia Detection & Classification:** Identifies X-ray images as Normal, Pneumonia, or Unknown using a pre-trained CNN model.
- **Visual Explainability (Grad-CAM):** Provides a visual heatmap overlay showing which regions of the X-ray the model focused on.
- **Desktop GUI:** Provides a user-friendly Tkinter desktop application for easy interaction.
- **Efficient and Reliable:** Assists medical professionals by providing fast and reliable initial detection.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Likith-2004/Chest-X-Ray-Pneumonia-Detection.git
cd Chest-X-Ray-Pneumonia-Detection
   
2. **Set up a Virtual Environment (Optional but Recommended)**:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
Download and Place the Model: Ensure pneumonia_unknown_model.pth is in the root directory of the project. This is the trained PyTorch CNN model for detecting pneumonia.


## Run the application:

   ```bash
   python app.py
```


Access the Application: The Tkinter desktop GUI will open automatically.

Upload an Image: Use the upload feature to test the model by uploading a chest X-ray image. The system will analyze the image and display the classification, confidence score, and the Grad-CAM heatmap.

## Model

The detection relies on a modified ResNet-18 CNN model. The model's weights are loaded from pneumonia_unknown_model.pth. It is specifically trained to classify X-ray images into Normal, Pneumonia, or Unknown using advanced deep learning techniques (PyTorch) to achieve accurate results.

## Technologies Used :

- **Python**: Core programming language.
- **Tkinter**: Library used for building the cross-platform desktop GUI.
- **PyTorch/Torchvision**: Frameworks used to develop, train, and deploy the CNN model.
- **OpenCV(cv2)/PIL(Pillow)**: Used for image processing, transformations, and generating the Grad-CAM visualization.

## Contributing
Contributions are welcome! Please follow these steps to contribute:


## Fork the repository :
- Create a new branch (git checkout -b feature-branch).
- Make your changes and commit them (git commit -m 'Add new feature').
- Push to the branch (git push origin feature-branch).
- Open a Pull Request.

## License
  This project is licensed under the MIT License. See the LICENSE file for details.
