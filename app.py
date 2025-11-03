import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO

# -----------------------------
# 1. Define your CNN Model
# -----------------------------
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
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# -----------------------------
# 2. Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load("mnist_cnn_best.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# -----------------------------
# 3. Define Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Digit Recognition AI", page_icon="🔢", layout="centered")
st.title("🧠 Digit Recognition AI (PyTorch GPU Ready)")
st.markdown("Upload a handwritten digit image (0–9) or draw below for instant recognition.")

# -----------------------------
# 5. Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Digit", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1, keepdim=True).item()

    st.subheader(f"🔮 Predicted Digit: **{pred}**")

# -----------------------------
# 6. Draw Canvas (Optional)
# -----------------------------
st.markdown("Or draw a digit below 👇")

try:
    from streamlit_drawable_canvas import st_canvas
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        drawn_image = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype("uint8"))
        img_tensor = transform(drawn_image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
        st.subheader(f"✍️ Predicted Digit: **{pred}**")

except ModuleNotFoundError:
    st.warning("To enable drawing, install the canvas: `pip install streamlit-drawable-canvas`")

# -----------------------------
# 7. Footer
# -----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + PyTorch. Model trained on MNIST.")
