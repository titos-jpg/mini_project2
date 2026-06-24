import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px

from essentials import load_data

# ==========================================
# Konfigurasi Halaman
# ==========================================

st.set_page_config(
    page_title="Visualisasi Dataset",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visualisasi Dataset Diabetes")

df = load_data()

# ==========================================
# Dataset
# ==========================================

st.header("Dataset")

st.dataframe(df, use_container_width=True)

# ==========================================
# Statistik
# ==========================================

st.header("Statistik Deskriptif")

st.dataframe(df.describe(), use_container_width=True)

# ==========================================
# Distribusi Outcome
# ==========================================

st.header("Distribusi Kelas Diabetes")

fig = plt.hist(
    df,
    x="Outcome",
    color="Outcome",
    text_auto=True
)

fig.update_layout(
    xaxis_title="Outcome",
    yaxis_title="Jumlah Data"
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Histogram
# ==========================================

st.header("Distribusi Setiap Fitur")

fitur = st.selectbox(
    "Pilih Fitur",
    df.columns[:-1]
)

fig = plt.histogram(
    df,
    x=fitur,
    nbins=30,
    color_discrete_sequence=["royalblue"]
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Heatmap Korelasi
# ==========================================

st.header("Heatmap Korelasi")

fig, ax = plt.subplots(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

st.pyplot(fig)

# ==========================================
# Scatter Plot
# ==========================================

st.header("Scatter Plot")

col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox(
        "Sumbu X",
        df.columns[:-1],
        index=1
    )

with col2:
    y_axis = st.selectbox(
        "Sumbu Y",
        df.columns[:-1],
        index=5
    )

fig = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color=df["Outcome"].astype(str),
    hover_data=df.columns
)

st.plotly_chart(fig, use_container_width=True)