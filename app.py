# ─────────────────────────────────────────────────────────────────
# CELL 13.2 — Streamlit App for Deployment
# ─────────────────────────────────────────────────────────────────

import streamlit as st
import joblib
import numpy as np

# --- Configuration (Ensure these match your training setup) ---
FEATURES = ['wc', 'CR', 'SR', 'CC', 'CFA', 'CCA', 'sfc', 'CS', 'TC']

# --- Load the Model and Scaler ---
# Ensure these files (gradient_boosting_model.joblib, standard_scaler.joblib)
# are in the same directory as your app.py when deployed, or provide correct paths.
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('gradient_boosting_model.joblib')
        scaler = joblib.load('standard_scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler files not found. Make sure 'gradient_boosting_model.joblib' and 'standard_scaler.joblib' are in the correct directory.")
        st.stop()

loaded_model, loaded_scaler = load_model_and_scaler()

# --- Define a Prediction Function ---
def predict_compressive_strength_app(mix_design_params: dict) -> float:
    input_features = [mix_design_params[feat] for feat in FEATURES]
    new_X = np.array(input_features).reshape(1, -1)
    # IMPORTANT: Gradient Boosting was trained on UNSEALED data (X_train), so no scaling here.
    # If using a scaled model, uncomment the line below:
    # new_X_processed = loaded_scaler.transform(new_X)
    new_X_processed = new_X
    prediction = loaded_model.predict(new_X_processed)[0]
    return prediction

# --- Streamlit UI ---
st.set_page_config(page_title="Rubberized Concrete Compressive Strength Predictor", layout="centered")
st.title("🧱 Rubberized Concrete Compressive Strength Predictor")
st.markdown("Predict the compressive strength of rubberized concrete based on mix design parameters.")

st.header("Input Mix Design Parameters")

# Input fields for each feature
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        wc = st.number_input('w/c Ratio', min_value=0.2, max_value=0.8, value=0.4, step=0.01)
        CR = st.number_input('Coarse Rubber (CR) (kg/m³)', min_value=0.0, max_value=500.0, value=50.0, step=1.0)
        SR = st.number_input('Fine Rubber (SR) (kg/m³)', min_value=0.0, max_value=30.0, value=5.0, step=0.1)

    with col2:
        CC = st.number_input('Cement Content (CC) (kg/m³)', min_value=200.0, max_value=700.0, value=400.0, step=1.0)
        CFA = st.number_input('Fine Aggregate (CFA) (kg/m³)', min_value=0.0, max_value=1500.0, value=700.0, step=1.0)
        CCA = st.number_input('Coarse Aggregate (CCA) (kg/m³)', min_value=0.0, max_value=1800.0, value=1100.0, step=1.0)

    with col3:
        sfc = st.number_input('Steel Fibre (sfc) (%)', min_value=0.0, max_value=25.0, value=1.0, step=0.1)
        CS = st.number_input('Curing Age (CS) (days)', min_value=1.0, max_value=365.0, value=28.0, step=1.0)
        TC = st.number_input('Curing Temp (TC) (°C)', min_value=0.0, max_value=100.0, value=20.0, step=0.1)

    submitted = st.form_submit_button("Predict Compressive Strength")

    if submitted:
        mix_design_params = {
            'wc': wc, 'CR': CR, 'SR': SR, 'CC': CC, 
            'CFA': CFA, 'CCA': CCA, 'sfc': sfc, 'CS': CS, 'TC': TC
        }
        
        predicted_fc = predict_compressive_strength_app(mix_design_params)
        
        # --- Visualizing the Result ---
        st.success(f"Predicted Compressive Strength: **{predicted_fc:.2f} MPa**")
        
        st.subheader("Mix Design Profile")
        
        # Normalize values slightly for the radar chart scale (0 to 1) 
        # based on the min/max you set in the number inputs
        categories = list(mix_design_params.keys())
        values = list(mix_design_params.values())
        
        # Simple normalization for visualization purposes
        # (This scales the inputs so they fit on one circular graph)
        max_vals = [0.8, 500, 30, 700, 1500, 1800, 25, 365, 100]
        norm_values = [v / m for v, m in zip(values, max_vals)]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=norm_values + [norm_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Current Mix'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Relative Parameter Distribution"
        )

        st.plotly_chart(fig)