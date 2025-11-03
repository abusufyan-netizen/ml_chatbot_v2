import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import os
from utils.helper import load_digit_model, preprocess_image

st.title("🖊️ Draw and Predict")

# Load model
try:
    model = load_digit_model()
except Exception as e:
    st.error(str(e))
    model = None

# Canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype('uint8'))
        processed = preprocess_image(img)
        if model is not None:
            pred = np.argmax(model.predict(processed))
            st.success(f"✅ Predicted Digit: **{pred}**")
            os.makedirs("data", exist_ok=True)
            df = pd.DataFrame([[datetime.datetime.now(), pred]], columns=["timestamp", "prediction"])
            df.to_csv("data/history.csv", mode='a', index=False, header=not os.path.exists("data/history.csv"))
        else:
            st.warning("Model not loaded. Train the model first.")
    else:
        st.warning("Please draw a digit before predicting.")
