#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- 1. Define ALL_FEATURES and categorical options as per your model's training ---
# This list MUST exactly match the 'feature_names_in_' attribute of your trained model.
# Based on the provided .pkl file content, these are the expected features:
ALL_FEATURES = [
    'Crop_Year', 'Area', 'Production', 'kharif_rainfall', 'rabi_rainfall',
    'summer_rainfall', 'yearly_rainfall', 'kharif_temp', 'rabi_temp',
    'summer_temp', 'yearly_temp', '_c0', 'N', 'P', 'K', 'pH', 'relevant_rainfall',
    'relevant_temperature', 'rainfall_temp_interaction', 'NPK_ratio',
    'lagged_production_1yr', 'lagged_yield_1yr', 'State_Encoded',
    'District_Encoded', 'Crop_Encoded',
    'Season_Encoded_0', 'Season_Encoded_1', 'Season_Encoded_2', 'Season_Encoded_3',
    'Season_Encoded_4', 'Season_Encoded_5', 'Drought_Risk'
]

# --- Define lists for Streamlit dropdowns (human-readable values) ---
# IMPORTANT: These lists should contain the unique, human-readable values
# that correspond to the numerical encodings in your model's training data.
# You MUST replace these with the actual unique values from your dataset.

# Example State Names (replace with your actual states from training data)
STATES = sorted([
    'Andaman and Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam',
    'Bihar', 'Chandigarh', 'Chhattisgarh', 'Dadra and Nagar Haveli', 'Goa',
    'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu and Kashmir', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
])

# Example Crop Names (replace with your actual crops from training data)
CROPS = sorted([
    'Rice', 'Wheat', 'Maize', 'Jowar(Sorghum)', 'Bajra', 'Cotton(lint)', 'Sugarcane',
    'Groundnut', 'Pulses total', 'Gram', 'Arhar/tur', 'Potato', 'Onion',
    'Tomato', 'Banana', 'Mango', 'Coconut', 'Apple', 'Barley(JAV)', 'Bhindi',
    'Brinjal', 'Cabbage', 'Carrot', 'Chilli (Dry)', 'Coriander', 'Cucumber',
    'Garlic', 'Ginger', 'Grapes', 'Lemon', 'Litchi', 'Mustard', 'Orange',
    'Papaya', 'Peas (vegetable)', 'Pineapple', 'Soyabean', 'Sweet Potato',
    'Tea', 'Tobacco', 'Turmeric', 'Urad', 'Yam'
])

# Season names (these align with Season_Encoded_0 to Season_Encoded_5)
SEASONS = ['autumn', 'kharif', 'rabi', 'summer', 'whole year', 'winter']

# --- Define mapping dictionaries for encoded features ---
# These mappings are CRUCIAL. You MUST populate these with the exact numerical
# encodings that your LabelEncoder or OrdinalEncoder produced during model training.
# For demonstration, I'm creating simple index-based mappings, but these are unlikely
# to match your actual model's encodings.

# Placeholder mapping for State_Encoded
# Example: If 'Andhra Pradesh' was encoded as 0, 'Assam' as 1, etc.
STATE_ENCODING_MAP = {state.lower().replace(' ', '_'): i for i, state in enumerate(STATES)}
# If your model's encoder for states is available, you would load it and use it here.
# e.g., with open('state_encoder.pkl', 'rb') as f: state_encoder = pickle.load(f)
# and then use state_encoder.transform([selected_state_cleaned])[0]

# Placeholder mapping for Crop_Encoded
# Example: If 'Rice' was encoded as 0, 'Wheat' as 1, etc.
CROP_ENCODING_MAP = {crop.lower().replace(' ', '_').replace('(', '').replace(')', ''): i for i, crop in enumerate(CROPS)}
# If your model's encoder for crops is available, you would load it and use it here.
# e.g., with open('crop_encoder.pkl', 'rb') as f: crop_encoder = pickle.load(f)
# and then use crop_encoder.transform([selected_crop_cleaned])[0]

# --- 2. Load the trained model ---
# Ensure 'random_forest_model.pkl' is in the same directory as this Streamlit script,
# or provide the full path to the file.
try:
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Error: 'random_forest_model.pkl' not found. Please ensure the model file is in the correct directory.")
    st.stop() # Stop the app if model is not found
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# --- 3. Streamlit UI Layout ---
st.set_page_config(page_title="🌾AGRIPREDICT", layout="centered")

st.title("🌾AGRIPREDICT")
st.markdown("ML-Driven Insights for Data-Driven Crop Production, Yield Forecasting, and Risk Mitigation")

# --- Input Fields ---
st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (in hectares)", min_value=0.0, value=1.0, format="%.2f")
    selected_season = st.selectbox("Season", SEASONS)
    relevant_rainfall = st.number_input("Relevant Rainfall (mm)", min_value=0.0, value=500.0, format="%.2f")
    
    
with col2:
     production = st.number_input("Production (previous, if applicable)", min_value=0.0, value=100.0, format="%.2f")
     selected_state = st.selectbox("State Name", STATES)
     relevant_temperature = st.number_input("Relevant Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0, format="%.2f")
     selected_crop = st.selectbox("Crop", CROPS)
    
   
    

# --- Prediction Button ---
if st.button("Get Prediction", help="Click to predict crop yield based on inputs"):
    # --- Prepare input for the model ---
    # Create a dictionary to hold all feature values, initialized to 0
    input_features = {feature: 0 for feature in ALL_FEATURES}

    # Populate the numerical features from UI inputs
    input_features['Area'] = area
    input_features['Production'] = production
    input_features['relevant_rainfall'] = relevant_rainfall
    input_features['relevant_temperature'] = relevant_temperature

    # --- Handle Encoded Categorical Features ---
    # State_Encoded
    selected_state_cleaned = selected_state.lower().replace(' ', '_').strip()
    if selected_state_cleaned in STATE_ENCODING_MAP:
        input_features['State_Encoded'] = STATE_ENCODING_MAP[selected_state_cleaned]
    else:
        st.error(f"Error: State '{selected_state}' not found in encoding map. Please check STATE_ENCODING_MAP.")
        st.stop()

    # Crop_Encoded
    selected_crop_cleaned = selected_crop.lower().replace(' ', '_').replace('(', '').replace(')', '').strip()
    if selected_crop_cleaned in CROP_ENCODING_MAP:
        input_features['Crop_Encoded'] = CROP_ENCODING_MAP[selected_crop_cleaned]
    else:
        st.error(f"Error: Crop '{selected_crop}' not found in encoding map. Please check CROP_ENCODING_MAP.")
        st.stop()

    # Season (one-hot encoded)
    # Reset all Season_Encoded_X to 0 first
    for i in range(6):
        input_features[f'Season_Encoded_{i}'] = 0
    # Set the correct encoded season based on selected_season
    if selected_season == 'autumn': input_features['Season_Encoded_0'] = 1
    elif selected_season == 'kharif': input_features['Season_Encoded_1'] = 1
    elif selected_season == 'rabi': input_features['Season_Encoded_2'] = 1
    elif selected_season == 'summer': input_features['Season_Encoded_3'] = 1
    elif selected_season == 'whole year': input_features['Season_Encoded_4'] = 1
    elif selected_season == 'winter': input_features['Season_Encoded_5'] = 1
    else:
        st.error(f"Error: Season '{selected_season}' not recognized. This should not happen with the dropdown.")
        st.stop()


    # --- Set default values for other features expected by the model but not in UI ---
    # These values are crucial for the model to work. Adjust them based on your data's typical values.
    input_features['Crop_Year'] = 2023 # Example: A recent year
    input_features['kharif_rainfall'] = input_features['relevant_rainfall'] * 0.6 # Example: Proportion of relevant
    input_features['rabi_rainfall'] = input_features['relevant_rainfall'] * 0.3
    input_features['summer_rainfall'] = input_features['relevant_rainfall'] * 0.1
    input_features['yearly_rainfall'] = input_features['relevant_rainfall'] # Sum of seasonal rainfall might be different
    input_features['kharif_temp'] = input_features['relevant_temperature'] + 5 # Example: Seasonal temperature variations
    input_features['rabi_temp'] = input_features['relevant_temperature'] - 5
    input_features['summer_temp'] = input_features['relevant_temperature'] + 10
    input_features['yearly_temp'] = input_features['relevant_temperature']
    input_features['_c0'] = 0 # Default for the mysterious _c0 feature
    input_features['N'] = 50 # Default NPK values (Nitrogen, Phosphorus, Potassium)
    input_features['P'] = 30
    input_features['K'] = 20
    input_features['pH'] = 6.5 # Default pH value
    input_features['rainfall_temp_interaction'] = input_features['relevant_rainfall'] * input_features['relevant_temperature'] / 100 # Example interaction term
    input_features['NPK_ratio'] = (input_features['N'] + input_features['P'] + input_features['K']) / 3 if (input_features['N'] + input_features['P'] + input_features['K']) > 0 else 0
    input_features['lagged_production_1yr'] = input_features['Production'] # Example: Assume last year's production is current
    input_features['lagged_yield_1yr'] = input_features['Production'] / input_features['Area'] if input_features['Area'] > 0 else 0 # Example: Calculate yield
    input_features['District_Encoded'] = 0 # Default for District_Encoded (since District is not in UI)
    input_features['Drought_Risk'] = 0 # Default for Drought_Risk (assuming no drought by default)


    # Convert the dictionary to a list in the exact order ALL_FEATURES
    # This is crucial for the model to receive features in the correct sequence.
    final_input_array = np.array([input_features[feature] for feature in ALL_FEATURES]).reshape(1, -1)

    # Convert to DataFrame as scikit-learn models often expect DataFrame input
    # This also ensures column names are correctly passed if the model expects them.
    input_df = pd.DataFrame(final_input_array, columns=ALL_FEATURES)

    # --- Make Prediction ---
    try:
        with st.spinner("Predicting..."):
            prediction = model.predict(input_df)[0] # Assuming single prediction output
        st.success("Prediction successful!")
        st.markdown(f"## Predicted Crop Yield: **{prediction:.2f} units**")
        st.info("Note: This prediction is based on the provided inputs and the loaded Random Forest model.")
        st.warning("Please ensure the `STATE_ENCODING_MAP` and `CROP_ENCODING_MAP` in the code are updated with your model's actual numerical encodings for accurate predictions.")
        st.warning("Also, review the default values for features not in the UI (e.g., `Crop_Year`, `N`, `P`, `K`, `_c0`, `yearly_rainfall`, `seasonal_temps`, etc.) to ensure they are appropriate for your model.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all inputs are valid and the model is compatible with the input format.")

st.markdown("---")
st.markdown("### How to run this application:")
st.markdown("1.  Save the code above as a Python file (e.g., `app.py`).")
st.markdown("2.  Ensure your `random_forest_model.pkl` file is in the same directory as `app.py`.")
st.markdown("3.  **Crucially, update the `STATE_ENCODING_MAP` and `CROP_ENCODING_MAP` dictionaries in the Python code with the exact numerical values that your model's original data encoding produced.**")
st.markdown("4.  Open your terminal or command prompt.")
st.markdown("5.  Navigate to the directory where you saved `app.py`.")
st.markdown("6.  Run the command: `streamlit run app.py`")
st.markdown("7.  A new tab will open in your web browser with the application.")

