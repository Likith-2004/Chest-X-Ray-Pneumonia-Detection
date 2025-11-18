<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>README - Chest X-Ray Pneumonia Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
            background-color: #f9f9f9;
        }
        h1 {
            color: #1e88e5;
            border-bottom: 3px solid #1e88e5;
            padding-bottom: 10px;
            margin-top: 0;
        }
        h2 {
            color: #43a047;
            border-bottom: 1px solid #ccc;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        h3 {
            color: #ef6c00;
            margin-top: 20px;
        }
        code {
            background-color: #eee;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: Consolas, monospace;
        }
        pre {
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        pre code {
            background-color: transparent;
            color: #ecf0f1;
            padding: 0;
        }
        ul {
            list-style-type: disc;
            margin-left: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: #fff;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .note {
            background-color: #fff3e0;
            border-left: 5px solid #ff9800;
            padding: 10px;
            margin: 15px 0;
        }
    </style>
</head>
<body>

    <h1>&#129505; Chest X-Ray Pneumonia Detector with Grad-CAM</h1>

    <p>This project provides a desktop application for classifying chest X-ray images to detect the presence of pneumonia. Built using <strong>Tkinter</strong> and <strong>PyTorch</strong>, the system not only classifies images but also incorporates <strong>Grad-CAM (Gradient-weighted Class Activation Mapping)</strong> to visualize the exact areas of the X-ray image that led to the model's decision, enhancing transparency and trust in the AI diagnosis.</p>

    <hr>

    <h2>✨ Features</h2>
    <ul>
        <li><strong>Pneumonia Classification:</strong> Classifies uploaded X-ray images into one of three categories:
            <ul>
                <li><strong>Normal</strong> (Green)</li>
                <li><strong>Pneumonia</strong> (Red)</li>
                <li><strong>Unknown</strong> (Orange - for uncertain or out-of-distribution inputs)</li>
            </ul>
        </li>
        <li><strong>Visual Explainability:</strong> Integrates <strong>Grad-CAM</strong> to generate a heatmap overlay, highlighting the region(s) of interest (ROI) the model focused on for its prediction.</li>
        <li><strong>Confidence Scoring:</strong> Displays the prediction confidence percentage for the classified class.</li>
        <li><strong>Desktop GUI:</strong> User-friendly interface built with the <code>tkinter</code> library.</li>
    </ul>

    <hr>

    <h2>🚀 Setup and Installation</h2>
    <p>To run this application, you will need Python and several essential libraries, including PyTorch and OpenCV.</p>

    <h3>1. Prerequisites</h3>
    <ul>
        <li>Python 3.x</li>
        <li>The trained PyTorch model file: <strong><code>pneumonia_unknown_model.pth</code></strong></li>
        <li>(Optional) A background image file: <strong><code>background.png</code></strong></li>
    </ul>

    <h3>2. Install Dependencies</h3>
    <p>Install the required libraries using pip:</p>
    <pre><code># Core Libraries
pip install torch torchvision numpy
# GUI and Image Processing Libraries
pip install pillow opencv-python
</code></pre>

    <h3>3. Place External Files</h3>
    <p>Ensure the following files are located in the <strong>same directory</strong> as your <code>app.py</code> script:</p>

    <table>
        <thead>
            <tr>
                <th>Filename</th>
                <th>Description</th>
                <th>Importance</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>app.py</code></td>
                <td>The main application code.</td>
                <td>Essential</td>
            </tr>
            <tr>
                <td><strong><code>pneumonia_unknown_model.pth</code></strong></td>
                <td>The trained ResNet-18 model weights.</td>
                <td><strong>Essential (App will exit if missing)</strong></td>
            </tr>
            <tr>
                <td><code>background.png</code></td>
                <td>Image for the Tkinter window background.</td>
                <td>Optional (App uses default color if missing)</td>
            </tr>
        </tbody>
    </table>

    <hr>

    <h2>▶️ Running the Application</h2>
    <p>Execute the Python script from your terminal:</p>
    <pre><code>python app.py
</code></pre>

    <h3>Usage</h3>
    <ol>
        <li>The Tkinter window will launch.</li>
        <li>Click the <strong>"&#128193; Upload X-Ray"</strong> button.</li>
        <li>Select an X-ray image file (<code>.png</code>, <code>.jpg</code>, or <code>.jpeg</code>).</li>
        <li>The left panel will display the <strong>Original Image</strong>.</li>
        <li>The right panel will display the <strong>Analysis Results</strong>:
            <ul>
                <li>The prediction and confidence score will be shown at the bottom.</li>
                <li>The Grad-CAM overlay image will appear, showing the heatmap where the red/yellow areas indicate the highest model attention.</li>
            </ul>
        </li>
    </ol>

    <hr>

    <h2>⚙️ Technical Details</h2>

    <table>
        <thead>
            <tr>
                <th>Component</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Model</strong></td>
                <td>ResNet-18 (pre-trained structure, custom classification head)</td>
            </tr>
            <tr>
                <td><strong>Input Size</strong></td>
                <td>Images are transformed and resized to $224 \times 224$.</td>
            </tr>
            <tr>
                <td><strong>Target Layer (Grad-CAM)</strong></td>
                <td><code>model.layer4[1].conv2</code> (The final convolutional layer before the global average pooling).</td>
            </tr>
            <tr>
                <td><strong>Device</strong></td>
                <td>Supports automatic selection of CUDA (GPU) or CPU.</td>
            </tr>
        </tbody>
    </table>

    <h3>&#x1f6d1; Troubleshooting</h3>
    <div class="note">
        <ul>
            <li><strong>"Error: Model file not found..."</strong>: This means <code>pneumonia_unknown_model.pth</code> is missing or misspelled. Check the file location and name.</li>
            <li><strong>Window opens but no background</strong>: The <code>background.png</code> file is missing. The application is functional but using a plain background.</li>
        </ul>
    </div>

    <hr>

    <h2>🤝 Contributing</h2>
    <p>We welcome contributions to improve the model accuracy, expand features, or enhance the GUI!</p>
    <ol>
        <li>Fork the repository.</li>
        <li>Create a feature branch.</li>
        <li>Submit a Pull Request.</li>
    </ol>

    <hr>

    <h2>📝 License</h2>
    <p>This project is licensed under the MIT License.</p>

</body>
</html>
