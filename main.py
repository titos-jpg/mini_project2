import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from essentials import load_data, load_model_ann, load_model_logistic

# Konfigurasi halaman
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# HEADER
# =========================
st.title("🩺 Diabetes Prediction System")

st.markdown("""
Selamat datang pada aplikasi **Prediksi Diabetes**.

Aplikasi ini dibangun menggunakan dua model Machine Learning yaitu:

- 📈 Linear Regression
- 🧠 Artificial Neural Network (ANN)

Selain melakukan prediksi, aplikasi ini juga menyediakan visualisasi data untuk membantu memahami karakteristik dataset diabetes.
""")

st.divider()

# =========================
# FITUR
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Prediction")
    st.write("""
    Halaman ini digunakan untuk memprediksi kemungkinan diabetes berdasarkan data pasien.
    
    Model yang digunakan:
    
    - Linear Regression
    - Artificial Neural Network
    """)

with col2:
    st.subheader("📊 Visualization")
    st.write("""
    Halaman ini menampilkan visualisasi dataset diabetes, seperti:

    - Distribusi data
    - Korelasi fitur
    - Histogram
    - Scatter Plot
    """)

st.divider()

# =========================
# INFORMASI DATASET
# =========================
st.subheader("📚 Dataset")

st.write("""
Dataset yang digunakan merupakan **Pima Indians Diabetes Dataset** yang terdiri dari beberapa fitur kesehatan pasien.

Fitur yang digunakan:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

Target:

- Outcome
    - 0 = Tidak Diabetes
    - 1 = Diabetes
""")

st.divider()

# =========================
# FOOTER
# =========================
st.success(
    "Silakan pilih menu **Prediction** atau **Visualization** pada sidebar sebelah kiri."
)