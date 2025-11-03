import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO
import pandas as pd
from utils.data_logger import save_feedback

# -----------------------------
# Model Definition
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

@st.cache_resource
def load_model():
    model = CNNModel()
    model.load_state_dict(torch.load("model/mnist_cnn_best.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# -----------------------------
# Streamlit UI Navigation
# -----------------------------
st.set_page_config(page_title="Digit Recognition AI", page_icon="🔢", layout="wide")

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["Predict", "History"])

# -----------------------------
# Page 1: Prediction
# -----------------------------
if page == "Predict":
    st.title("🧠 Digit Recognition AI")
    st.markdown("Upload or draw a handwritten digit (0–9).")

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    canvas_result = None

    try:
        from streamlit_drawable_canvas import st_canvas
        st.markdown("Or draw a digit below 👇")
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
    except ModuleNotFoundError:
        st.warning("To enable drawing, install with: `pip install streamlit-drawable-canvas`")

    img = None
    if uploaded_file:
        img = Image.open(uploaded_file).convert("L")
        st.image(img, caption="Uploaded Digit", use_container_width=True)
    elif canvas_result and canvas_result.image_data is not None:
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype("uint8"))

    if img is not None:
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()

        st.subheader(f"🔮 Predicted Digit: **{pred}**")

        # Feedback Section
        st.markdown("### 🧩 Is this prediction correct?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Correct"):
                save_feedback(img, predicted=pred, correct=pred)
                st.success("Feedback saved as correct ✅")

        with col2:
            correct_digit = st.text_input("If wrong, enter correct digit (0–9):", "")
            if st.button("❌ Wrong Prediction"):
                if correct_digit.isdigit() and 0 <= int(correct_digit) <= 9:
                    save_feedback(img, predicted=pred, correct=int(correct_digit))
                    st.error(f"Feedback saved: Correct digit is {correct_digit}")
                else:
                    st.warning("Please enter a valid digit (0–9).")

# -----------------------------
# Page 2: History
# -----------------------------
elif page == "History":
    st.title("📊 Prediction History")
    import os

    if os.path.exists("data/logs/feedback_log.csv"):
        df = pd.read_csv("data/logs/feedback_log.csv")
        if not df.empty:
            df["Color"] = df["is_correct"].map({1: "✅", 0: "❌"})
            st.dataframe(df[["timestamp", "predicted", "correct", "Color"]])
            accuracy = (df["is_correct"].mean()) * 100 if df["is_correct"].count() > 0 else 0
            st.metric("Model Accuracy (based on feedback)", f"{accuracy:.2f}%")
        else:
            st.info("No feedback history yet.")
    else:
        st.info("No feedback history found.")
