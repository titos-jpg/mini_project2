from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Prediksi Diabetes",
    layout="wide",
    initial_sidebar_state="expanded",
)


FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FEATURE_LABELS = {
    "Pregnancies": "Jumlah Kehamilan",
    "Glucose": "Glukosa",
    "BloodPressure": "Tekanan Darah",
    "SkinThickness": "Ketebalan Kulit",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Usia",
}

MODEL_OPTIONS = {
    "Logistic Regression": {
        "model": "logistic_regression_model.pkl",
        "scaler": "ml_scaler.joblib",
        "type": "sklearn",
    },
    "Artificial Neural Network": {
        "model": "ann_model.keras",
        "scaler": "ann_scaler.joblib",
        "type": "keras",
    },
}


def project_root():
    return Path(__file__).resolve().parents[1]


@st.cache_data
def load_reference_data():
    data_path = project_root() / "data" / "diabetes.csv"
    if data_path.exists():
        return pd.read_csv(data_path)

    return pd.DataFrame(columns=FEATURES + ["Outcome"])


@st.cache_resource
def load_sklearn_model(model_filename, scaler_filename):
    from joblib import load

    models_path = project_root() / "models"
    model = load(models_path / model_filename)
    scaler = load(models_path / scaler_filename)
    using_fallback = False

    if not hasattr(model, "predict"):
        model, scaler = train_logistic_model_from_data()
        using_fallback = True

    return model, scaler, using_fallback


@st.cache_resource
def train_logistic_model_from_data():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler

    data_path = project_root() / "data" / "diabetes.csv"
    df = pd.read_csv(data_path)

    x = df[FEATURES].astype(float).to_numpy()
    y = df["Outcome"].astype(int).to_numpy()

    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_scaled, y)

    return model, scaler


@st.cache_resource
def load_keras_model(model_filename, scaler_filename):
    from joblib import load
    from tensorflow import keras

    models_path = project_root() / "models"
    model = keras.models.load_model(models_path / model_filename, compile=False)
    scaler = load(models_path / scaler_filename)
    return model, scaler


def get_feature_defaults(df):
    defaults = {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 20,
        "Insulin": 80,
        "BMI": 30.0,
        "DiabetesPedigreeFunction": 0.4,
        "Age": 30,
    }

    if df.empty:
        return defaults

    for feature in FEATURES:
        defaults[feature] = float(df[feature].median())

    return defaults


def build_input_form(defaults):
    st.subheader("Data Pasien")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                FEATURE_LABELS["Pregnancies"],
                min_value=0,
                max_value=20,
                value=int(defaults["Pregnancies"]),
                step=1,
            )
            glucose = st.number_input(
                FEATURE_LABELS["Glucose"],
                min_value=0,
                max_value=250,
                value=int(defaults["Glucose"]),
                step=1,
            )
            blood_pressure = st.number_input(
                FEATURE_LABELS["BloodPressure"],
                min_value=0,
                max_value=140,
                value=int(defaults["BloodPressure"]),
                step=1,
            )
            skin_thickness = st.number_input(
                FEATURE_LABELS["SkinThickness"],
                min_value=0,
                max_value=100,
                value=int(defaults["SkinThickness"]),
                step=1,
            )

        with col2:
            insulin = st.number_input(
                FEATURE_LABELS["Insulin"],
                min_value=0,
                max_value=900,
                value=int(defaults["Insulin"]),
                step=1,
            )
            bmi = st.number_input(
                FEATURE_LABELS["BMI"],
                min_value=0.0,
                max_value=70.0,
                value=float(round(defaults["BMI"], 1)),
                step=0.1,
            )
            pedigree = st.number_input(
                FEATURE_LABELS["DiabetesPedigreeFunction"],
                min_value=0.0,
                max_value=3.0,
                value=float(round(defaults["DiabetesPedigreeFunction"], 3)),
                step=0.001,
                format="%.3f",
            )
            age = st.number_input(
                FEATURE_LABELS["Age"],
                min_value=1,
                max_value=120,
                value=int(defaults["Age"]),
                step=1,
            )

        submitted = st.form_submit_button("Prediksi")

    patient_data = pd.DataFrame(
        [
            {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree,
                "Age": age,
            }
        ],
        columns=FEATURES,
    )

    return submitted, patient_data


def predict_probability(model, scaler, patient_data, model_type):
    input_values = patient_data[FEATURES].astype(float).to_numpy()
    scaled_data = scaler.transform(input_values)

    if model_type == "keras":
        prediction = model.predict(scaled_data, verbose=0)
        prediction_values = prediction.reshape(-1)
        if len(prediction_values) > 1:
            return float(prediction_values[-1])
        return float(prediction_values[0])

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(scaled_data)[0][1])

    return float(model.predict(scaled_data)[0])


def render_prediction(probability, threshold):
    prediction = int(probability >= threshold)
    percentage = probability * 100

    st.subheader("Hasil Prediksi")

    result_col, prob_col = st.columns([1, 2])

    with result_col:
        if prediction == 1:
            st.error("Terdeteksi berisiko diabetes")
        else:
            st.success("Tidak terdeteksi diabetes")

    with prob_col:
        st.metric("Probabilitas Diabetes", f"{percentage:.2f}%")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Threshold prediksi: {threshold:.2f}")


def render_patient_preview(patient_data):
    preview = patient_data.rename(columns=FEATURE_LABELS)
    st.dataframe(preview, use_container_width=True, hide_index=True)


def show_prediction():
    st.title("Prediksi Diabetes")
    st.write(
        "Masukkan data pasien, pilih model, lalu jalankan prediksi untuk "
        "melihat estimasi risiko diabetes."
    )
    st.caption(
        "Hasil prediksi adalah estimasi dari model machine learning dan bukan "
        "pengganti diagnosis tenaga medis."
    )

    df = load_reference_data()
    defaults = get_feature_defaults(df)

    st.sidebar.header("Pengaturan Model")
    selected_model = st.sidebar.selectbox("Pilih model", list(MODEL_OPTIONS))
    threshold = st.sidebar.slider(
        "Threshold klasifikasi",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    submitted, patient_data = build_input_form(defaults)

    st.subheader("Preview Input")
    render_patient_preview(patient_data)

    if not submitted:
        st.info("Klik tombol Prediksi setelah data pasien diisi.")
        return

    model_config = MODEL_OPTIONS[selected_model]

    try:
        if model_config["type"] == "keras":
            model, scaler = load_keras_model(
                model_config["model"],
                model_config["scaler"],
            )
        else:
            model, scaler, using_fallback = load_sklearn_model(
                model_config["model"],
                model_config["scaler"],
            )
            if using_fallback:
                st.warning(
                    "File logistic_regression_model.pkl saat ini berisi DataFrame, "
                    "bukan model. Prediksi Logistic Regression dijalankan memakai "
                    "model fallback yang dilatih ulang dari data/diabetes.csv."
                )

        probability = predict_probability(
            model,
            scaler,
            patient_data,
            model_config["type"],
        )
        render_prediction(probability, threshold)

    except ModuleNotFoundError as error:
        st.error(
            "Dependency model belum terpasang. Jalankan "
            "`pip install -r requirements.txt`, lalu coba lagi."
        )
        st.caption(f"Detail: {error}")
    except FileNotFoundError as error:
        st.error("File model atau scaler tidak ditemukan di folder models.")
        st.caption(f"Detail: {error}")
    except Exception as error:
        st.error("Prediksi gagal dijalankan.")
        st.caption(f"Detail: {type(error).__name__}: {error}")


show_prediction()
