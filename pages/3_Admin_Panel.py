import streamlit as st
import os

st.title("⚙️ Admin Panel")

if os.path.exists("data/history.csv"):
    st.info("Manage your app history here.")
    if st.button("🧹 Clear History"):
        os.remove("data/history.csv")
        st.success("Prediction history cleared successfully.")
else:
    st.warning("No history file found yet.")
