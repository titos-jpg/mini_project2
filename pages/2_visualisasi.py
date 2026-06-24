from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Visualisasi Dataset",
    layout="wide",
    initial_sidebar_state="expanded",
)


REQUIRED_COLUMNS = {
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
}

OUTCOME_LABELS = {
    0: "Tidak Diabetes",
    1: "Diabetes",
}

COLOR_MAP = {
    "Tidak Diabetes": "#2ca02c",
    "Diabetes": "#d62728",
}


def load_visualization_data():
    local_path = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"
    if local_path.exists():
        return pd.read_csv(local_path)

    try:
        from essentials import load_data

        df = load_data()
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    return None


def validate_dataset(df):
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        st.error(
            "Dataset tidak memiliki kolom yang dibutuhkan: "
            + ", ".join(sorted(missing_columns))
        )
        return False

    return True


def prepare_data(df):
    prepared = df.copy()
    prepared["OutcomeLabel"] = prepared["Outcome"].map(OUTCOME_LABELS)
    return prepared


def filter_data(df, numeric_features):
    st.sidebar.header("Filter Data")

    selected_outcomes = st.sidebar.multiselect(
        "Status diabetes",
        options=list(OUTCOME_LABELS.values()),
        default=list(OUTCOME_LABELS.values()),
    )

    filtered = df[df["OutcomeLabel"].isin(selected_outcomes)]

    range_features = ["Age", "Glucose", "BMI"]
    for feature in range_features:
        min_value = float(df[feature].min())
        max_value = float(df[feature].max())
        selected_range = st.sidebar.slider(
            feature,
            min_value=min_value,
            max_value=max_value,
            value=(min_value, max_value),
        )
        filtered = filtered[
            (filtered[feature] >= selected_range[0])
            & (filtered[feature] <= selected_range[1])
        ]

    st.sidebar.caption(f"{len(filtered)} dari {len(df)} data ditampilkan")
    return filtered[numeric_features + ["Outcome", "OutcomeLabel"]]


def render_summary(df):
    total_data = len(df)
    diabetes_count = int((df["Outcome"] == 1).sum())
    non_diabetes_count = int((df["Outcome"] == 0).sum())
    diabetes_rate = (diabetes_count / total_data * 100) if total_data else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data", total_data)
    col2.metric("Diabetes", diabetes_count)
    col3.metric("Tidak Diabetes", non_diabetes_count)
    col4.metric("Persentase Diabetes", f"{diabetes_rate:.1f}%")


def render_target_distribution(df):
    target_count = (
        df["OutcomeLabel"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Jumlah")
    )

    fig = px.bar(
        target_count,
        x="Status",
        y="Jumlah",
        color="Status",
        text="Jumlah",
        color_discrete_map=COLOR_MAP,
        title="Distribusi Target",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Jumlah Data")
    st.plotly_chart(fig, use_container_width=True)


def render_feature_distribution(df, numeric_features):
    col1, col2 = st.columns([1, 1])

    with col1:
        selected_feature = st.selectbox(
            "Pilih fitur untuk histogram",
            numeric_features,
            index=numeric_features.index("Glucose"),
        )

    with col2:
        bin_count = st.slider("Jumlah bin", min_value=5, max_value=60, value=30)

    fig_hist = px.histogram(
        df,
        x=selected_feature,
        color="OutcomeLabel",
        nbins=bin_count,
        marginal="box",
        opacity=0.75,
        barmode="overlay",
        color_discrete_map=COLOR_MAP,
        title=f"Distribusi {selected_feature}",
    )
    fig_hist.update_layout(
        xaxis_title=selected_feature,
        yaxis_title="Jumlah Data",
        legend_title_text="Status",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_box = px.box(
        df,
        x="OutcomeLabel",
        y=selected_feature,
        color="OutcomeLabel",
        points="all",
        color_discrete_map=COLOR_MAP,
        title=f"Sebaran {selected_feature} Berdasarkan Outcome",
    )
    fig_box.update_layout(
        xaxis_title="Status",
        yaxis_title=selected_feature,
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)


def render_scatter(df, numeric_features):
    col1, col2, col3 = st.columns(3)

    with col1:
        x_axis = st.selectbox(
            "Sumbu X",
            numeric_features,
            index=numeric_features.index("Glucose"),
        )

    with col2:
        y_axis = st.selectbox(
            "Sumbu Y",
            numeric_features,
            index=numeric_features.index("BMI"),
        )

    with col3:
        size_feature = st.selectbox(
            "Ukuran titik",
            ["Tidak digunakan"] + numeric_features,
            index=0,
        )

    fig = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        color="OutcomeLabel",
        size=None if size_feature == "Tidak digunakan" else size_feature,
        hover_data=numeric_features + ["OutcomeLabel"],
        color_discrete_map=COLOR_MAP,
        title=f"Hubungan {x_axis} dan {y_axis}",
    )
    fig.update_layout(legend_title_text="Status")
    st.plotly_chart(fig, use_container_width=True)


def render_correlation(df, numeric_features):
    selected_features = st.multiselect(
        "Pilih fitur untuk korelasi",
        options=numeric_features + ["Outcome"],
        default=["Glucose", "BMI", "Age", "Pregnancies", "Outcome"],
    )

    if len(selected_features) < 2:
        st.info("Pilih minimal dua fitur untuk menampilkan heatmap korelasi.")
        return

    corr = df[selected_features].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Heatmap Korelasi",
    )
    fig.update_layout(coloraxis_colorbar_title="Korelasi")
    st.plotly_chart(fig, use_container_width=True)


def show_visualization():
    st.title("Visualisasi Dataset Diabetes")
    st.write(
        "Gunakan filter dan pilihan chart untuk mengeksplorasi pola pada "
        "dataset Pima Indians Diabetes."
    )

    df = load_visualization_data()

    if df is None:
        st.error(
            "Gagal memuat dataset. Pastikan file data/diabetes.csv tersedia "
            "atau fungsi load_data() di essentials.py berjalan."
        )
        return

    if not validate_dataset(df):
        return

    numeric_features = [column for column in df.columns if column != "Outcome"]
    df = prepare_data(df)
    filtered_df = filter_data(df, numeric_features)

    if filtered_df.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini.")
        return

    tab_summary, tab_distribution, tab_scatter, tab_correlation, tab_data = st.tabs(
        ["Ringkasan", "Distribusi", "Hubungan", "Korelasi", "Data"]
    )

    with tab_summary:
        render_summary(filtered_df)
        render_target_distribution(filtered_df)

    with tab_distribution:
        render_feature_distribution(filtered_df, numeric_features)

    with tab_scatter:
        render_scatter(filtered_df, numeric_features)

    with tab_correlation:
        render_correlation(filtered_df, numeric_features)

    with tab_data:
        st.dataframe(filtered_df, use_container_width=True)
        st.download_button(
            "Download data terfilter",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="diabetes_filtered.csv",
            mime="text/csv",
        )


show_visualization()
