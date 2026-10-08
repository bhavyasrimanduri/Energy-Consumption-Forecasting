import streamlit as st
import pandas as pd
from pycaret.regression import setup, compare_models, tune_model, predict_model

st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecasting")
st.write("AutoML-based household energy consumption prediction")

DATA_FILE = "household_energy_consumption.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

@st.cache_resource
def train_model():
    df = load_data().copy()

    df["Date"] = pd.to_datetime(df["Date"])

    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    df.drop(columns=["Date", "Household_ID"], inplace=True)

    setup(
        data=df,
        target="Energy_Consumption_kWh",
        session_id=42,
        verbose=False
    )

    best_model = compare_models()
    tuned_model = tune_model(best_model)

    return tuned_model


try:
    df = load_data()

    st.success("Dataset loaded successfully!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        st.metric("Households", df["Household_ID"].nunique())

    with col3:
        st.metric(
            "Average Consumption",
            f"{df['Energy_Consumption_kWh'].mean():.2f} kWh"
        )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Train AutoML Model")

    if st.button("🚀 Train Model", type="primary"):

        with st.spinner("Training AutoML model... This may take a few minutes."):

            model = train_model()

        st.session_state["model"] = model

        st.success("Model trained successfully!")

    if "model" in st.session_state:

        st.divider()

        st.header("🔮 Energy Consumption Prediction")

        model = st.session_state["model"]

        col1, col2 = st.columns(2)

        with col1:

            household_size = st.number_input(
                "Household Size",
                min_value=1,
                max_value=20,
                value=4
            )

            temperature = st.number_input(
                "Average Temperature (°C)",
                min_value=-20.0,
                max_value=60.0,
                value=25.0
            )

            has_ac = st.selectbox(
                "Has AC?",
                ["No", "Yes"]
            )

        with col2:

            peak_usage = st.number_input(
                "Peak Hours Usage (kWh)",
                min_value=0.0,
                max_value=100.0,
                value=3.0
            )

            date = st.date_input(
                "Date"
            )

        if st.button("🔮 Predict Energy Consumption"):

            prediction_date = pd.Timestamp(date)

            input_data = pd.DataFrame({
                "Household_Size": [household_size],
                "Avg_Temperature_C": [temperature],
                "Has_AC": [has_ac],
                "Peak_Hours_Usage_kWh": [peak_usage],
                "Month": [prediction_date.month],
                "DayOfWeek": [prediction_date.dayofweek]
            })

            prediction = predict_model(
                model,
                data=input_data
            )

            predicted_value = prediction["prediction_label"].iloc[0]

            st.success(
                f"Predicted Energy Consumption: {predicted_value:.2f} kWh"
            )

            st.metric(
                "Predicted Consumption",
                f"{predicted_value:.2f} kWh"
            )

except FileNotFoundError:
    st.error(
        "household_energy_consumption.csv was not found. "
        "Make sure the CSV is in the same folder as app.py."
    )

except Exception as e:
    st.error("An error occurred while running the application.")
    st.exception(e)