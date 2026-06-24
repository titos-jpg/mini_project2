import pandas as pd
import streamlit as st
from joblib import load
import tensorflow as tf
from tensorflow import keras

@st.cache_data
def load_data():
    return pd.read_csv(
        'https://raw.githubusercontent.com/titos-jpg/mini_project2/refs/heads/main/diabetes.csv                                                 '
    )

@st.cache_resource
def load_model_ann():
    try:
        model_Ann = keras.models.load_model('mini_project2/model_ann.h5')
        scaler_Ann = load('mini_project2/scaler_ann.joblib')
        return model_Ann, scaler_Ann
    except Exception as e:
        st.error(f"terdapat error pada pemanggilan ANN: {e}")
        return None, None
    
@st.cache_resource
def load_model_logistic():
    try:
        model_Logistic = load('mini_project2/model_logistic.joblib')
        scaler_Logistic = load('mini_project2/scaler_logistic.joblib')
        return model_Logistic, scaler_Logistic
    except Exception as e:
        st.error(f"terdapat error pada pemanggilan Logistic Regression: {e}")
        return None, None