import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "pneumonia_unknown_model.pth"
CLASSES = ["Normal", "Pneumonia", "Unknown"]

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[:, target_class].backward()
        weights = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max()+1e-8)
        return cam.cpu().numpy(), target_class

gradcam = GradCAM(model, model.layer4[1].conv2)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if not file_path: 
        return

    img = Image.open(file_path).convert("RGB")
    img_tk = ImageTk.PhotoImage(img.resize((250,250)))
    lbl_img.config(image=img_tk)
    lbl_img.image = img_tk

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item() * 100

    lbl_result.config(
        text=f"{CLASSES[pred]} ({confidence:.2f}%)",
        fg="red" if CLASSES[pred]=="Pneumonia" else ("green" if CLASSES[pred]=="Normal" else "orange"),
        font=("Helvetica",16,"bold")
    )

    # Grad-CAM visualization
    cam, _ = gradcam.generate(input_tensor, pred)
    cam = cv2.resize(cam, (224,224))
    img_np = np.array(img.resize((224,224)))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)/255
    overlay = 0.4*heatmap + np.float32(img_np)/255
    overlay = overlay / np.max(overlay)
    overlay_img = Image.fromarray(np.uint8(255*overlay))

    overlay_tk = ImageTk.PhotoImage(overlay_img.resize((250,250)))
    lbl_gradcam.config(image=overlay_tk)
    lbl_gradcam.image = overlay_tk

window = tk.Tk()
window.title("Chest X-Ray Pneumonia Detector + Grad-CAM")
window.geometry("650x600")
window.configure(bg="#f0f0f0")

btn_upload = Button(window, text="📂 Upload X-Ray", command=open_image, 
                    font=("Helvetica",12,"bold"), bg="#3b6a91", fg="white", relief="raised")
btn_upload.pack(pady=15)

lbl_img = Label(window, bg="#f0f0f0")
lbl_img.pack()

lbl_result = Label(window, text="Prediction:", font=("Helvetica",14,"bold"), bg="#f0f0f0")
lbl_result.pack(pady=10)

lbl_gradcam = Label(window, bg="#d9f4f7")
lbl_gradcam.pack(pady=15)

window.mainloop()