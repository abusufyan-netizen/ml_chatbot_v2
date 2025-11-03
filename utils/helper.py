import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

MODEL_PATH = "model/digit_model.h5"

def load_digit_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("⚠️ Model not found. Train it first using train_model.py")
    return load_model(MODEL_PATH)

def preprocess_image(img):
    img = img.convert('L').resize((28, 28))
    img = 255 - np.array(img)
    img = img / 255.0
    img = img.reshape(1, 28, 28)
    return img
