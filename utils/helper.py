# utils/helper.py

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from PIL import Image

# ------------------------------------------------------------
# Define the CNN architecture (same as used during training)
# ------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ------------------------------------------------------------
# Model Loading
# ------------------------------------------------------------
def load_pytorch_model(model_path="model/mnist_cnn_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel()
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ Model loaded successfully on {device}")
        return model, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, device

# ------------------------------------------------------------
# Image Preprocessing for MNIST (28x28 grayscale)
# ------------------------------------------------------------
def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = image.convert("L").resize((28, 28))
    else:
        raise TypeError("Input must be a PIL.Image")

    img_array = np.array(image, dtype=np.float32)
    img_array = 255 - img_array  # invert colors if background is white
    img_array = img_array / 255.0
    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    return tensor

# ------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------
def predict_digit(model, device, image):
    try:
        tensor = preprocess_image(image).to(device)
        with torch.no_grad():
            output = model(tensor)
            pred = output.argmax(dim=1, keepdim=True)
            confidence = torch.exp(output.max()).item()
        return pred.item(), confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0
