import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the Saved Model and Artifacts ---
# NOTE: Ensure 'house_price_model.pkl' is in the same directory as this app.py file
try:
    # Load the entire pipeline (which includes the preprocessor and the model)
    model = joblib.load("gbr.pkl")
except FileNotFoundError:
    st.error("Error: 'house_price_model.pkl' not found. Please ensure your saved model file is in the same directory as app.py.")
    st.stop()

# --- 2. Streamlit UI and Input Handling ---
st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏡 House Price Prediction App")
st.markdown("Use the sliders and input fields below to estimate a home's sale price.")

# Define input fields based on the features used in your model
col1, col2, col3 = st.columns(3)

with col1:
    lotarea = st.number_input(
        "Lot Area (SqFt)",
        min_value=100,
        max_value=200000,
        value=8000,
        help="The size of the lot in square feet."
    )

with col2:
    overallqual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=7,
        help="Rates the overall material and finish of the house (1=Poor, 10=Excellent)."
    )

with col3:
    yearbuilt = st.number_input(
        "Year Built",
        min_value=1800,
        max_value=2025,
        value=2003,
        help="Original construction date."
    )

# --- Add a Prediction Button and Logic ---
if st.button("Predict Sale Price", type="primary"):
    # 1. Create a DataFrame from the user inputs
    input_data = pd.DataFrame({
        'LotArea': [lotarea],
        'OverallQual': [overallqual],
        'YearBuilt': [yearbuilt],
        # If you used other features like 'Neighborhood', 'HouseStyle', etc., 
        # you MUST include them here, even with placeholder/default values.
    })

    try:
        # 2. Make the prediction using the loaded model (which handles preprocessing)
        # The model predicts the log of the price (implied by regression models on house prices)
        log_prediction = model.predict(input_data)[0]
        
        # 3. Convert prediction back from log scale to dollars (Inverse Transformation)
        # Assuming the target variable (SalePrice) was log-transformed (np.log or np.log1p)
        
        # NOTE: Using np.exp for inverse of log, or np.expm1 for inverse of log1p
        predicted_price = np.expm1(log_prediction)
        
        # 4. Display the result
        st.success("---")
        st.metric(
            label="Estimated Sale Price",
            value=f"${predicted_price:,.0f}",
            delta="Predicted using Ridge Regression"
        )
        st.success("---")
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the input features match the features your model was trained on, including case and data type.")
