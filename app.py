
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Thalassemia Prediction App", layout="centered")

# --- Load the model bundle ---
try:
    # Adjust path if your .pkl file is not in the same directory as app.py
    bundle = joblib.load("thalassemia_model_bundle.pkl")
    model = bundle['model']
    scaler = bundle['scaler']
    feature_order = bundle['feature_order']
    target_map = bundle['target_map']
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: thalassemia_model_bundle.pkl not found. Please ensure the file is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- App Title and Description ---
st.title("Thalassemia Prediction")
st.markdown("This application predicts the presence of Thalassemia based on blood parameters.")
st.markdown("--- ")

# --- Input Form ---
st.header("Patient Blood Parameters")
st.write("Please enter the values for the following blood parameters:")

# Create input fields for each feature in the feature_order
input_data = {}
# Provide some reasonable default or min/max values
default_values = {
    'hbg': 12.0, 'mcv': 80.0, 'mch': 27.0, 'mchc': 33.0,
    'rbc': 4.5, 'rdw': 13.0, 'hba2': 2.5, 'hbf': 0.5
}
min_values = {
    'hbg': 5.0, 'mcv': 50.0, 'mch': 15.0, 'mchc': 20.0,
    'rbc': 2.0, 'rdw': 10.0, 'hba2': 0.0, 'hbf': 0.0
}
max_values = {
    'hbg': 18.0, 'mcv': 120.0, 'mch': 40.0, 'mchc': 40.0,
    'rbc': 7.0, 'rdw': 25.0, 'hba2': 10.0, 'hbf': 10.0
}

for feature in feature_order:
    input_data[feature] = st.number_input(
        f"**{feature.upper()}**",
        min_value=float(min_values.get(feature, 0.0)),
        max_value=float(max_values.get(feature, 100.0)),
        value=float(default_values.get(feature, 0.0)),
        step=0.1,
        format="%.2f"
    )

# --- Prediction Button ---
if st.button("Predict Thalassemia"):
    # Convert input data to a numpy array in the correct order
    features_array = np.array([[input_data[f] for f in feature_order]])

    # Scale the features
    scaled_features = scaler.transform(features_array)

    # Make prediction
    prediction_label = None
    prediction_proba = None

    try:
        prediction_raw = model.predict(scaled_features)[0]
        prediction_label = target_map.get(prediction_raw, "Unknown")

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(scaled_features)[0]
            prediction_proba = {target_map.get(i, f"Class_{i}"): p for i, p in enumerate(proba)}

        st.subheader("Prediction Result")
        if prediction_label == "Thalassemia":
            st.error(f"**Prediction: {prediction_label}**")
        else:
            st.success(f"**Prediction: {prediction_label}**")

        if prediction_proba:
            st.write("--- ")
            st.subheader("Prediction Probabilities")
            for label, prob in prediction_proba.items():
                st.write(f"- {label}: {prob:.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("--- ")
st.caption("Note: This is a demo application for educational purposes. Consult a medical professional for diagnosis.")
'''

with open("app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)

print("Streamlit app (app.py) created successfully!")
