import streamlit as st

st.set_page_config(page_title="Digit Recognition App", page_icon="🔢", layout="wide")
st.markdown('<style>body{background:#0e1117;color:#fafafa}</style>', unsafe_allow_html=True)

st.title("🔢 Handwritten Digit Recognition (Dark Mode)")
st.markdown("""
Welcome to the **AI-powered digit recognition app**!  
Use the left sidebar to navigate through:
- 🖊️ Draw and Predict  
- 📊 Prediction History  
- ⚙️ Admin Panel
""")
st.image("https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png", use_container_width=True)
