
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Appliance Energy Prediction",
    page_icon="⚡",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("appliance_energy_model(2).pkl")

model = load_model()

# Title
st.title("⚡ Appliance Energy Consumption Prediction")

st.write(
    "Enter the temperature to predict the appliance's energy consumption."
)

st.divider()

# Input
temperature = st.number_input(
    "Temperature (°C)",
    min_value=-20.0,
    max_value=60.0,
    value=25.0,
    step=0.1
)

# Prediction
if st.button("Predict Energy Consumption"):

    # Input must have the same column name used during model training
    input_data = pd.DataFrame({
        "Temperature (°C)": [temperature]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Energy Consumption: {prediction:.2f} kWh"
    )

    st.subheader("Prediction Details")

    result = pd.DataFrame({
        "Temperature (°C)": [temperature],
        "Energy Consumption (kWh)": [round(prediction, 2)]
    })

    st.dataframe(
        result,
        use_container_width=True
    )

st.divider()

st.caption(
    "Machine Learning Based Appliance Energy Consumption Prediction"
)
