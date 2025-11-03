# 🔢 Digit Recognition App (Dark Mode)

A Streamlit app to draw handwritten digits (0–9) and predict them using a TensorFlow model trained on MNIST.

## Quick Start

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional, will create model/digit_model.h5):
   ```bash
   python train_model.py
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. Use the sidebar to navigate pages: Draw and Predict, Prediction History, Admin Panel.

## Notes
- This repo runs fully locally; no Drive integration included.
- History is stored in `data/history.csv`.
