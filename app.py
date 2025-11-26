import streamlit as st
import numpy as np
import pickle


with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🏠 House Price Prediction App")
st.write("Enter the details of the house below to predict its price. All units are indicated next to each input.")


median_income = st.slider('Median Income (in $10,000s)', 0.0, 15.0, 3.0, 0.1)
house_age = st.slider('House Age (years)', 0, 50, 20)
average_rooms = st.slider('Average Rooms per House', 1.0, 10.0, 5.0, 0.1)
average_bedrooms = st.slider('Average Bedrooms per House', 1.0, 5.0, 2.0, 0.1)
population = st.slider('Population of the Area', 50, 5000, 1000)
average_occupancy = st.slider('Average Occupancy per House', 1.0, 5.0, 3.0, 0.1)
latitude = st.slider('Latitude', 32.0, 42.0, 34.0, 0.01)
longitude = st.slider('Longitude', -124.0, -114.0, -118.0, 0.01)

if st.button("Predict Price"):
    input_data = np.array([[median_income, house_age, average_rooms, average_bedrooms,
                            population, average_occupancy, latitude, longitude]])
    predicted_price = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${predicted_price*100000:.2f}")  # multiply by 100,000 to get USD
else:
    st.info("Enter house details and click 'Predict Price' to see the prediction.")

