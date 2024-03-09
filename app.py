# import streamlit as st
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model

# # Title
# st.title('Rain Prediction for the Next Day')
# st.markdown("""
# Predict whether it will rain tomorrow by entering today's weather data. 
# Fill in the details below and click the **Predict Rainfall** button.
# """)

# # Sidebar inputs
# year = st.number_input('Year', min_value=2000, max_value=2030, value=2011)
# doy = st.number_input('Day of Year', min_value=1, max_value=366, value=25)
# temp = st.number_input('Temperature (°C)', value=24.31)
# pressure = st.number_input('Pressure', value=963.87)
# rel_hum = st.number_input('Relative Humidity (%)', value=26.38)
# wind_dir = st.number_input('Wind Direction (degrees)', value=126.13)
# wind_speed = st .number_input('Wind Speed (m/s)', value=1.05)

# # Load the scaler
# scaler = joblib.load('C:/Users/Boniface/Weather-Forecast/scaler_wf.save')  # Update this path

# # Load the model
# model = load_model('C:/Users/Boniface/Weather-Forecast/my_model_wf.h5')  # Update this path

# # Button to make prediction
# if st.button('Predict Rainfall'):
#     # Prepare input data
#     input_features = np.array([[year, doy, temp, pressure, rel_hum, wind_dir, wind_speed]])
#     input_features_scaled = scaler.transform(input_features)

#     # Make prediction
#     prediction = model.predict(input_features_scaled)
#     will_rain_next_day = prediction[0][0] > 0.5

#     # Display result
#     if will_rain_next_day:
#         st.success("The model predicts rain for the next day. 🌧️")
#     else:
#         st.success("The model predicts no rain for the next day. ☀️")


import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Title and Introduction
st.title('Rain Prediction for the Next Day')
st.markdown("""
Predict whether it will rain tomorrow by entering today's weather data. 
Fill in the details below and click the **Predict Rainfall** button.
""")

# Layout for inputs
col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input('Year', min_value=2000, max_value=2030, value=2011)
    doy = st.number_input('Day of Year', min_value=1, max_value=366, value=25)

with col2:
    temp = st.number_input('Temperature (°C)', value=24.31)
    pressure = st.number_input('Pressure', value=963.87)

with col3:
    rel_hum = st.number_input('Relative Humidity (%)', value=26.38)
    wind_dir = st.number_input('Wind Direction (degrees)', value=126.13)
    wind_speed = st.number_input('Wind Speed (m/s)', value=1.05)

# Placeholder for the prediction output
prediction_placeholder = st.empty()

# Load the scaler and model
scaler = joblib.load('scaler_wf.save')
model = load_model('my_model_wf.h5')

# Predict button
if st.button('Predict Rainfall'):
    # Prepare input data
    input_features = np.array([[year, doy, temp, pressure, rel_hum, wind_dir, wind_speed]])
    input_features_scaled = scaler.transform(input_features)

    # Make prediction
    prediction = model.predict(input_features_scaled)
    will_rain_next_day = prediction[0][0] > 0.5

    # Display result in the placeholder
    if will_rain_next_day:
        prediction_placeholder.success("The model predicts rain for the next day. 🌧️")
    else:
        prediction_placeholder.success("The model predicts no rain for the next day. ☀️")
