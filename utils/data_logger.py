import os
import csv
from datetime import datetime
from PIL import Image

# Ensure directories exist
os.makedirs("data/feedback", exist_ok=True)
os.makedirs("data/logs", exist_ok=True)

FEEDBACK_CSV = "data/logs/feedback_log.csv"

# Initialize CSV if not exists
if not os.path.exists(FEEDBACK_CSV):
    with open(FEEDBACK_CSV, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "image_path", "predicted", "correct", "is_correct"])

def save_feedback(image, predicted, correct=None):
    """Save user feedback (correct or incorrect)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f"data/feedback/{timestamp}.png"
    image.save(image_path)

    is_correct = 1 if correct == predicted else 0 if correct else ""
    with open(FEEDBACK_CSV, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, image_path, predicted, correct, is_correct])
    
    return image_path
