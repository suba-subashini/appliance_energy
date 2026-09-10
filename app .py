
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Appliance Energy Prediction",
    page_icon="⚡",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("appliance_energy_model.pkl")

model = load_model()

st.title("⚡ Appliance Energy Consumption Prediction")

st.write(
    "Enter the temperature to predict appliance energy consumption."
)

st.divider()

temperature = st.number_input(
    "Enter Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

if st.button("Predict Energy Consumption"):

    input_data = pd.DataFrame(
        {"Temperature (°C)": [temperature]}
    )

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Energy Consumption: {prediction:.2f} kWh"
    )

    result = pd.DataFrame({
        "Temperature (°C)": [temperature],
        "Energy Consumption (kWh)": [round(prediction, 2)]
    })

    st.dataframe(result, use_container_width=True)

st.divider()
st.caption("Appliance Energy Prediction using Machine Learning")
