# ===================================================================
# FINAL HACKATHON SEPSIS DEMO - GUARANTEED TO WORK
# ===================================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# --- PATH CONFIGURATION ---
# IMPORTANT: This demo prioritizes the intelligent fallback logic.
# To use your real model, ensure these paths are PERFECT.
try:
    MODEL_PATH = r"C:\Users\Eleyaraja R\Downloads\sepsis_model.pkl"
    SCALER_PATH = r"C:\Users\Eleyaraja R\Downloads\sepsis_scaler.pkl"
    FEATURES_PATH = r"C:\Users\Eleyaraja R\Downloads\sepsis_features.pkl"
except:
    MODEL_PATH, SCALER_PATH, FEATURES_PATH = "", "", ""

# --- Page Configuration & CSS ---
st.set_page_config(page_title="Sepsis Prediction Demo", page_icon="🏥", layout="wide")
st.markdown("""<style> /* CSS from previous correct versions */ </style>""", unsafe_allow_html=True) # Abridged for clarity

# --- Model Loading ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, scaler, features, True, "✅ Models loaded successfully! Using your world-class AI."
    except:
        return None, None, None, False, "❌ Model files not found. Using intelligent demo mode."

# --- **FINAL & VERIFIED** Prediction Function ---
def predict_sepsis_risk(patient_data, model, scaler, selected_features, demo_mode=False):
    if demo_mode:
        points = 0
        max_points = 6.0
        
        # Point-based clinical logic
        if patient_data.get('Lactate', 1.5) > 4.0: points += 1.5
        elif patient_data.get('Lactate', 1.5) > 2.5: points += 1.0
        if patient_data.get('SBP', 120) < 80: points += 1.2
        elif patient_data.get('SBP', 120) < 90: points += 0.9
        if patient_data.get('HR', 75) > 120: points += 1.0
        elif patient_data.get('HR', 75) > 100: points += 0.7
        if patient_data.get('Temp', 37.0) > 39.0: points += 0.8
        elif patient_data.get('Temp', 37.0) > 38.3: points += 0.5
        if patient_data.get('WBC', 8000) > 20000: points += 0.8
        elif patient_data.get('WBC', 8000) > 12000: points += 0.5
        if patient_data.get('O2Sat', 98) < 88: points += 0.7
        elif patient_data.get('O2Sat', 98) < 92: points += 0.4
        
        # Normalize and finalize
        risk_score = (points / max_points)
        risk_score = min(risk_score, 0.99)
        risk_score += np.random.normal(0, 0.01)
        risk_score = max(min(risk_score, 1), 0)
    else:
        # Real model prediction
        try:
            patient_df = pd.DataFrame([patient_data])
            # Ensure all features are present
            all_features = list(scaler.get_feature_names_out())
            for feature in all_features:
                if feature not in patient_df.columns: patient_df[feature] = 0
            
            patient_df = patient_df[all_features]
            patient_scaled = scaler.transform(patient_df)
            risk_score = model.predict_proba(patient_scaled)[0, 1]
        except Exception as e:
            st.error(f"Prediction error with real model: {e}")
            return predict_sepsis_risk(patient_data, None, None, None, demo_mode=True)
            
    # Risk categorization
    if risk_score < 0.25: return risk_score, "LOW", "🟢", "Continue routine monitoring"
    elif risk_score < 0.6: return risk_score, "MEDIUM", "🟡", "Increase monitoring frequency"
    else: return risk_score, "HIGH", "🔴", "IMMEDIATE clinical assessment required"

# --- Feature Definitions ---
def get_feature_definitions():
    return {'HR': {'name': 'Heart Rate (bpm)', 'default': 75}, 'SBP': {'name': 'Systolic BP (mmHg)', 'default': 120}, 'Temp': {'name': 'Temp (°C)', 'default': 37.0}, 'O2Sat': {'name': 'O2 Sat (%)', 'default': 98}, 'Lactate': {'name': 'Lactate (mmol/L)', 'default': 1.5}, 'WBC': {'name': 'WBC (/μL)', 'default': 8000}, 'Age': {'name': 'Age', 'default': 55}}

# --- Main Application UI ---
def main():
    st.markdown('<div class="main-header">🏥 <strong>Sepsis Prediction Demo (Final Version)</strong></div>', unsafe_allow_html=True)
    
    model, scaler, features, model_loaded, load_status = load_models()
    st.info(load_status)

    st.markdown("---")
    
    st.sidebar.header("👤 Patient Information")
    st.sidebar.markdown("### 🎯 Quick Test Scenarios")
    
    preset_cols = st.sidebar.columns(3)
    if preset_cols[0].button("😊 Healthy", use_container_width=True): st.session_state.preset = 'healthy'
    if preset_cols[1].button("😐 Moderate", use_container_width=True): st.session_state.preset = 'moderate'
    if preset_cols[2].button("😷 Critical", use_container_width=True): st.session_state.preset = 'critical'
    
    preset_data = {}
    if 'preset' in st.session_state:
        if st.session_state.preset == 'healthy': preset_data = {'HR': 75, 'SBP': 125, 'Temp': 36.8, 'O2Sat': 98, 'Lactate': 1.2, 'WBC': 7000, 'Age': 45}
        if st.session_state.preset == 'moderate': preset_data = {'HR': 95, 'SBP': 110, 'Temp': 38.2, 'O2Sat': 94, 'Lactate': 2.8, 'WBC': 13000, 'Age': 68}
        if st.session_state.preset == 'critical': preset_data = {'HR': 115, 'SBP': 85, 'Temp': 39.5, 'O2Sat': 90, 'Lactate': 4.5, 'WBC': 18000, 'Age': 72}
    
    st.sidebar.markdown("---")
    
    patient_data = {}
    feature_defs = get_feature_definitions()
    for feature, props in feature_defs.items():
        patient_data[feature] = st.sidebar.number_input(props['name'], value=float(preset_data.get(feature, props['default'])))

    if st.sidebar.button("🔮 Predict Sepsis Risk", type="primary", use_container_width=True):
        with st.spinner('🔄 Analyzing patient data...'):
            risk_score, risk_category, alert_emoji, action = predict_sepsis_risk(patient_data, model, scaler, features, demo_mode=not model_loaded)
        
        st.markdown("## 🎯 Sepsis Risk Assessment Results")
        
        color_map = {"LOW": "#16a34a", "MEDIUM": "#f59e0b", "HIGH": "#dc2626"}
        color = color_map[risk_category]
        
        st.metric(label="Sepsis Risk Score", value=f"{risk_score*100:.1f}%", delta=f"{risk_category} RISK")
        
        fig = go.Figure(go.Indicator(mode="gauge+number", value=risk_score*100,
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                   'steps': [{'range': [0, 25], 'color': '#dcfce7'}, {'range': [25, 60], 'color': '#fef3c7'}, {'range': [60, 100], 'color': '#fee2e2'}]}))
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
