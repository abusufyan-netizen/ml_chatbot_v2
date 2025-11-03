import streamlit as st
import pandas as pd
import os

st.title("📊 Prediction History")

if os.path.exists("data/history.csv"):
    df = pd.read_csv("data/history.csv")
    st.dataframe(df)
else:
    st.warning("No prediction history available yet.")
