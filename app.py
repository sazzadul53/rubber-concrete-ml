import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration ---
FEATURES = ['wc', 'CR', 'SR', 'CC', 'CFA', 'CCA', 'sfc', 'CS', 'TC']

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('gradient_boosting_model.joblib')
        scaler = joblib.load('standard_scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler files not found.")
        st.stop()

loaded_model, loaded_scaler = load_model_and_scaler()

def predict_compressive_strength_app(mix_design_params: dict) -> float:
    input_features = [mix_design_params[feat] for feat in FEATURES]
    new_X = np.array(input_features).reshape(1, -1)
    # Model trained on unscaled data as per README
    prediction = loaded_model.predict(new_X)[0]
    return prediction

# --- Streamlit UI ---
st.set_page_config(page_title="Rubberized Concrete Predictor", layout="wide")
st.title("🧱 Rubberized Concrete Strength Analysis")

st.header("Input Mix Parameters")

# Main page form using columns for a clean layout
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        wc = st.number_input('w/c Ratio', 0.2, 0.8, 0.4)
        CR = st.number_input('Coarse Rubber (kg/m³)', 0.0, 500.0, 50.0)
        SR = st.number_input('Fine Rubber (kg/m³)', 0.0, 30.0, 5.0)

    with col2:
        CC = st.number_input('Cement Content (kg/m³)', 200.0, 700.0, 400.0)
        CFA = st.number_input('Fine Aggregate (kg/m³)', 0.0, 1500.0, 700.0)
        CCA = st.number_input('Coarse Aggregate (kg/m³)', 0.0, 1800.0, 1100.0)

    with col3:
        sfc = st.number_input('Steel Fibre (%)', 0.0, 25.0, 1.0)
        CS = st.number_input('Curing Age (days)', 1.0, 365.0, 28.0)
        TC = st.number_input('Curing Temp (°C)', 0.0, 100.0, 20.0)
    
    submitted = st.form_submit_button("Analyze Mix Design")

# --- Results & Visualization ---
if submitted:
    params = {'wc': wc, 'CR': CR, 'SR': SR, 'CC': CC, 'CFA': CFA, 'CCA': CCA, 'sfc': sfc, 'CS': CS, 'TC': TC}
    prediction = predict_compressive_strength_app(params)
    
    # 1. Metric Display
    st.metric(label="Predicted Compressive Strength", value=f"{prediction:.2f} MPa")
    
    st.divider() # Adds a nice horizontal line to separate inputs from graphs
    
    # 2. Create columns for the graphs so they sit side-by-side
    graph_col1, graph_col2 = st.columns(2)
    
    with graph_col1:
        st.subheader("Mix Design Profile")
        
        # 1. Map the short variable names to full presentation labels
        display_names = {
            'wc': 'w/c Ratio',
            'CR': 'Coarse Rubber (CR)',
            'SR': 'Fine Rubber (SR)',
            'CC': 'Cement Content (CC)',
            'CFA': 'Fine Aggregate (CFA)',
            'CCA': 'Coarse Aggregate (CCA)',
            'sfc': 'Steel Fibre (sfc)',
            'CS': 'Curing Age (CS)',
            'TC': 'Curing Temp (TC)'
        }
        
        # 2. Apply the full names to the chart categories
        categories = [display_names[k] for k in params.keys()]
        values = list(params.values())
        
        # Normalizing values for the radar chart scale
        max_vals = [0.8, 500, 30, 700, 1500, 1800, 25, 365, 100]
        norm_values = [v / m for v, m in zip(values, max_vals)]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=norm_values + [norm_values[0]], 
            theta=categories + [categories[0]], 
            fill='toself',
            line_color='#2E86C1'
        ))
        
        # 3. Increased left/right margins (l=80, r=80) so the longer labels don't get cut off
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
            showlegend=False,
            margin=dict(t=40, b=40, l=80, r=80) 
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with graph_col2:
        st.subheader("Sensitivity Analysis: w/c vs Strength")
        wc_range = np.linspace(0.2, 0.8, 20)
        trends = [predict_compressive_strength_app({**params, 'wc': w}) for w in wc_range]
        
        fig_line = px.line(x=wc_range, y=trends, labels={'x': 'Water-Cement Ratio', 'y': 'Strength (MPa)'})
        fig_line.add_vline(x=wc, line_dash="dash", line_color="red", annotation_text="Current Mix")
        fig_line.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        st.plotly_chart(fig_line, use_container_width=True)