import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

# Page configurations
st.set_page_config(
    page_title="Marine Engine Predictive Maintenance System",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to render HTML cleanly without Streamlit converting indented lines to markdown code blocks
def render_html(html_code):
    cleaned_lines = [line.strip() for line in html_code.split('\n') if line.strip()]
    cleaned_html = "\n".join(cleaned_lines)
    st.markdown(cleaned_html, unsafe_allow_html=True)

# Fault Definitions (Exact dataset Fault_Label classes 0-7)
FAULT_CLASSES = {
    0: {
        "name": "Normal Operation",
        "desc": "All propulsion systems operating within nominal baseline parameters.",
        "color": "#10b981", # Green
        "severity": "Healthy",
        "rec": "Continue nominal operations. Execute standard engine room logbook entries. Next scheduled maintenance in 250 operational hours."
    },
    1: {
        "name": "Fuel Delivery System Anomaly",
        "desc": "Fuel flow rate is disproportionately high relative to engine load and shaft RPM.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Isolate fuel supply line. Inspect primary fuel filter elements for clogging. Test fuel injectors 1-4 for spray patterns and pressure loss."
    },
    2: {
        "name": "Low Cylinder Compression Pressure",
        "desc": "Piston ring or valve pressure leakage detected across combustion chambers.",
        "color": "#f59e0b", # Orange
        "severity": "Warning",
        "rec": "Schedule static compression test. Inspect intake/exhaust valve clearances. Check crankcase blow-by gas pressure."
    },
    3: {
        "name": "Combustion Heat / Exhaust Gas Anomaly",
        "desc": "Abnormal exhaust gas temperature differential across cylinder outlets.",
        "color": "#f59e0b", # Orange
        "severity": "Warning",
        "rec": "Check jacket cooling water flow rate and outlet temperature. Inspect exhaust manifold thermowells and clean turbocharger inlet screen."
    },
    4: {
        "name": "Radial Engine Vibration Fault",
        "desc": "Excessive radial vibration displacement recorded in X and Y sensor channels.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Reduce engine load immediately. Perform laser shaft alignment check. Inspect main bearing clearances and engine mount dampers."
    },
    5: {
        "name": "Lubrication System Thermal Anomaly",
        "desc": "Elevated oil sump temperature accompanied by declining feed line oil pressure.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Verify lube oil cooler seawater circulation flow. Inspect thermostatic control valve. Sample oil for viscosity and flashpoint analysis."
    },
    6: {
        "name": "Air Intake Pressure / Turbocharger Fault",
        "desc": "Boost air pressure drop detected under high engine load demands.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Inspect turbocharger compressor wheel and turbine blades. Check air intake filter differential pressure and charge air cooler cleanliness."
    },
    7: {
        "name": "Lubrication Pressure & Axial Vibration Fault",
        "desc": "Critical drop in oil feed pressure coupled with severe Z-axis (axial) thrust vibration.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "SHUT DOWN ENGINE IMMEDIATELY. Conduct crankcase sump inspection for metallic debris. Inspect main thrust bearing collar and oil pump."
    }
}

# 18 Sensor Feature Columns in exact training order
feature_cols = [
    'Shaft_RPM', 'Engine_Load', 'Fuel_Flow', 'Air_Pressure', 'Ambient_Temp', 'Oil_Temp', 'Oil_Pressure',
    'Vibration_X', 'Vibration_Y', 'Vibration_Z', 'Cylinder1_Pressure', 'Cylinder1_Exhaust_Temp',
    'Cylinder2_Pressure', 'Cylinder2_Exhaust_Temp', 'Cylinder3_Pressure', 'Cylinder3_Exhaust_Temp',
    'Cylinder4_Pressure', 'Cylinder4_Exhaust_Temp'
]

# Baseline Telemetry Default Values
DEFAULT_SENSOR_VALUES = {
    'Shaft_RPM': 960.0,
    'Engine_Load': 75.0,
    'Fuel_Flow': 130.0,
    'Air_Pressure': 1.15,
    'Ambient_Temp': 27.0,
    'Oil_Temp': 78.0,
    'Oil_Pressure': 3.4,
    'Vibration_X': 0.06,
    'Vibration_Y': 0.05,
    'Vibration_Z': 0.07,
    'Cylinder1_Pressure': 145.0,
    'Cylinder1_Exhaust_Temp': 420.0,
    'Cylinder2_Pressure': 145.0,
    'Cylinder2_Exhaust_Temp': 420.0,
    'Cylinder3_Pressure': 145.0,
    'Cylinder3_Exhaust_Temp': 420.0,
    'Cylinder4_Pressure': 145.0,
    'Cylinder4_Exhaust_Temp': 420.0,
}

# Custom Styling (Ultra-Modern Dark Glassmorphism)
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global setting */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #0f172a 0%, #020617 100%) !important;
        color: #f3f4f6 !important;
        font-family: 'Inter', sans-serif;
    }

    /* Dark Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #070a12 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        background-color: transparent !important;
        padding: 0.6rem 0.9rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(56, 189, 248, 0.1) !important;
        color: #38bdf8 !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    
    /* Universal Selectbox & Dropdown styling fix */
    div[data-baseweb="select"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] * {
        color: #f8fafc !important;
        background-color: transparent !important;
    }
    div[data-baseweb="popover"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #0f172a !important;
    }
    div[data-baseweb="popover"] li {
        color: #f8fafc !important;
        background-color: #0f172a !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
    }
    
    /* Adjust Streamlit padding */
    .block-container {
        padding-top: 3.2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Main Title Styling */
    .main-title {
        font-family: 'Inter', sans-serif !important;
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.3;
        margin-bottom: 0.2rem;
        word-wrap: break-word;
        letter-spacing: -0.02em;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* Ultra-Modern Glassmorphism Cards style */
    .white-card {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        color: #f9fafb !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        margin-bottom: 1rem !important;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .white-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 35px -10px rgba(56, 189, 248, 0.15) !important;
    }
    .white-card h3 {
        margin-top: 0 !important;
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 750 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    .white-card p.card-value {
        color: #f8fafc !important;
        font-size: 1.75rem !important;
        font-weight: 800 !important;
        margin: 0.4rem 0 0 0 !important;
    }
    .white-card p.card-desc {
        color: #64748b !important;
        font-size: 0.82rem !important;
        margin: 0.4rem 0 0 0 !important;
        font-weight: 500 !important;
    }

    /* Workflow layout */
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1.8rem;
        margin-bottom: 2rem;
        width: 100%;
        gap: 0.8rem;
    }
    .workflow-step {
        background: rgba(15, 23, 42, 0.75);
        color: #f9fafb;
        border-radius: 16px;
        padding: 1.3rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        width: 22%;
        min-height: 125px;
        transition: all 0.2s ease;
    }
    .workflow-step:hover {
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-3px);
    }
    .workflow-step h4 {
        margin: 0.5rem 0 0.3rem 0;
        color: #ffffff;
        font-size: 1.0rem;
        font-weight: 750;
    }
    .workflow-step p {
        margin: 0;
        color: #94a3b8;
        font-size: 0.82rem;
        line-height: 1.4;
    }
    .workflow-arrow {
        font-size: 1.8rem;
        color: #38bdf8;
        font-weight: bold;
        opacity: 0.8;
    }

    /* Action buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.6rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(56, 189, 248, 0.5) !important;
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
    }

    /* Subsections and subheaders */
    .section-header {
        font-size: 1.35rem;
        font-weight: 750;
        color: #ffffff;
        margin-top: 1.8rem;
        margin-bottom: 1.0rem;
        border-left: 4px solid #38bdf8;
        padding-left: 0.7rem;
    }
    
    /* Input elements */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* Status badge pill */
    .status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
</style>
""")

# Model Loading Asset Function
@st.cache_resource
def load_ml_assets():
    model_files = {
        'decision_tree': 'decision_tree_model.pkl',
        'svm': 'svm_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'xgboost': 'xgboost_model.pkl'
    }
    scaler_path = "scaler.pkl"
    metrics_path = "model_metrics.pkl"
    
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
            
    models = {}
    for key, filename in model_files.items():
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                models[key] = pickle.load(f)
                
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            metrics = pickle.load(f)
            
    return models, scaler, metrics

models, scaler, metrics_payload = load_ml_assets()

# Model Definitions
MODEL_OPTIONS = [
    "Decision Tree",
    "Support Vector Machine (SVM)",
    "Random Forest",
    "XGBoost"
]

MODEL_KEY_MAP = {
    "Decision Tree": "decision_tree",
    "Support Vector Machine (SVM)": "svm",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost"
}
MODEL_NAME_MAP = {v: k for k, v in MODEL_KEY_MAP.items()}

# Initialize Session State values for 18 features, model choice & prediction history
def init_session_state():
    for k, v in DEFAULT_SENSOR_VALUES.items():
        if f"input_{k}" not in st.session_state:
            st.session_state[f"input_{k}"] = v
            
    if 'active_model_key' not in st.session_state:
        st.session_state['active_model_key'] = 'decision_tree'
    if 'active_model_name' not in st.session_state:
        st.session_state['active_model_name'] = 'Decision Tree'
        
    if 'latest_pred' not in st.session_state:
        st.session_state['latest_pred'] = 0
    if 'latest_conf' not in st.session_state:
        st.session_state['latest_conf'] = 0.982
    if 'latest_health' not in st.session_state:
        st.session_state['latest_health'] = 98.5
    if 'latest_probs' not in st.session_state:
        st.session_state['latest_probs'] = np.array([0.982, 0.003, 0.002, 0.003, 0.002, 0.003, 0.002, 0.003])
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []

init_session_state()

# Helper function: Check individual sensor telemetry status
def check_sensor_status(key, val):
    if key == 'Shaft_RPM':
        if 850 <= val <= 1050: return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")
        elif 750 <= val <= 1100: return ("WARNING", "#f59e0b", "rgba(245, 158, 11, 0.15)")
        else: return ("CRITICAL", "#ef4444", "rgba(239, 68, 68, 0.15)")
    elif key == 'Engine_Load':
        if 40 <= val <= 88: return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")
        elif 25 <= val <= 98: return ("WARNING", "#f59e0b", "rgba(245, 158, 11, 0.15)")
        else: return ("CRITICAL", "#ef4444", "rgba(239, 68, 68, 0.15)")
    elif key == 'Fuel_Flow':
        if 80 <= val <= 150: return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")
        elif 60 <= val <= 170: return ("WARNING", "#f59e0b", "rgba(245, 158, 11, 0.15)")
        else: return ("CRITICAL", "#ef4444", "rgba(239, 68, 68, 0.15)")
    elif key == 'Air_Pressure':
        if 0.9 <= val <= 1.4: return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")
        elif 0.6 <= val <= 1.5: return ("WARNING", "#f59e0b", "rgba(245, 158, 11, 0.15)")
        else: return ("CRITICAL", "#ef4444", "rgba(239, 68, 68, 0.15)")
    elif key == 'Oil_Temp':
        if 65 <= val <= 90: return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")
        elif 60 <= val <= 105: return ("WARNING", "#f59e0b", "rgba(245, 158, 11, 0.15)")
        else: return ("CRITICAL", "#ef4444", "rgba(239, 68, 68, 0.15)")
    elif key == 'Oil_Pressure':
        if 2.5 <= val <= 4.5: return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")
        elif 1.8 <= val <= 5.0: return ("WARNING", "#f59e0b", "rgba(245, 158, 11, 0.15)")
        else: return ("CRITICAL", "#ef4444", "rgba(239, 68, 68, 0.15)")
    elif key in ['Vibration_X', 'Vibration_Y', 'Vibration_Z']:
        if val <= 0.15: return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")
        elif val <= 0.35: return ("WARNING", "#f59e0b", "rgba(245, 158, 11, 0.15)")
        else: return ("CRITICAL", "#ef4444", "rgba(239, 68, 68, 0.15)")
    elif 'Pressure' in key:
        if 130 <= val <= 170: return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")
        elif 100 <= val <= 180: return ("WARNING", "#f59e0b", "rgba(245, 158, 11, 0.15)")
        else: return ("CRITICAL", "#ef4444", "rgba(239, 68, 68, 0.15)")
    elif 'Exhaust_Temp' in key:
        if 350 <= val <= 480: return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")
        elif 300 <= val <= 550: return ("WARNING", "#f59e0b", "rgba(245, 158, 11, 0.15)")
        else: return ("CRITICAL", "#ef4444", "rgba(239, 68, 68, 0.15)")
    return ("NORMAL", "#10b981", "rgba(16, 185, 129, 0.15)")

# Core Model Prediction Function (Route input to specific trained ML model)
def predict_with_model(model_key, input_vector):
    if scaler is None or model_key not in models:
        st.error(f"Error: Model '{model_key}' or scaler file not loaded properly.")
        return 0, 0.0, 95.0, np.zeros(8)
        
    scaled_vector = scaler.transform([input_vector])
    model_obj = models[model_key]
    
    pred_label = int(model_obj.predict(scaled_vector)[0])
    
    if hasattr(model_obj, 'predict_proba'):
        try:
            pred_probs = model_obj.predict_proba(scaled_vector)[0]
            confidence_score = float(pred_probs[pred_label])
        except Exception:
            pred_probs = np.zeros(8)
            pred_probs[pred_label] = 1.0
            confidence_score = 0.95
    else:
        pred_probs = np.zeros(8)
        pred_probs[pred_label] = 1.0
        confidence_score = 0.90
        
    # Dynamic Engine Health Calculation based on prediction severity & model confidence
    if pred_label == 0:
        health_score = round(min(100.0, 88.0 + (confidence_score * 11.5)), 1)
    elif pred_label in [2, 3]:
        health_score = round(max(50.0, 78.0 - (confidence_score * 25.0)), 1)
    else:
        health_score = round(max(5.0, 48.0 - (confidence_score * 40.0)), 1)
        
    return pred_label, confidence_score, health_score, pred_probs

# Execute live prediction pipeline for active model
def run_prediction_pipeline(log_history=False):
    input_vector = [st.session_state[f"input_{col}"] for col in feature_cols]
    active_key = st.session_state.get('active_model_key', 'decision_tree')
    
    pred_label, confidence_score, health_score, pred_probs = predict_with_model(active_key, input_vector)
    
    st.session_state['latest_pred'] = pred_label
    st.session_state['latest_conf'] = confidence_score
    st.session_state['latest_health'] = health_score
    st.session_state['latest_probs'] = pred_probs
    st.session_state['latest_model_key'] = active_key
    st.session_state['latest_model_name'] = MODEL_NAME_MAP.get(active_key, active_key)
    
    if log_history:
        fault_name = FAULT_CLASSES[pred_label]['name']
        severity = FAULT_CLASSES[pred_label]['severity']
        timestamp_str = datetime.now().strftime("%H:%M:%S")
        model_disp = MODEL_NAME_MAP.get(active_key, active_key)
        
        history_entry = {
            "Time": timestamp_str,
            "Model": model_disp,
            "Predicted Fault": fault_name,
            "Confidence": f"{confidence_score:.1%}",
            "Engine Health": f"{health_score:.1f}%",
            "Severity": severity
        }
        
        hist = st.session_state['prediction_history']
        if not hist or hist[0]['Time'] != timestamp_str or hist[0]['Model'] != model_disp:
            st.session_state['prediction_history'].insert(0, history_entry)
            st.session_state['prediction_history'] = st.session_state['prediction_history'][:10]
            
    return pred_label, confidence_score, health_score, pred_probs

# Preset telemetry loading logic (Updates telemetry sliders genuinely without forcing prediction)
def load_preset(scenario_name):
    for k, v in DEFAULT_SENSOR_VALUES.items():
        st.session_state[f"input_{k}"] = v

    if scenario_name == "Fuel Delivery System Anomaly":
        st.session_state['input_Fuel_Flow'] = 188.0
        st.session_state['input_Engine_Load'] = 45.0
        st.session_state['input_Shaft_RPM'] = 820.0
    elif scenario_name == "Low Cylinder Compression Pressure":
        st.session_state['input_Cylinder1_Pressure'] = 88.0
        st.session_state['input_Cylinder2_Pressure'] = 92.0
        st.session_state['input_Cylinder3_Pressure'] = 86.0
        st.session_state['input_Cylinder4_Pressure'] = 89.0
    elif scenario_name == "Combustion Heat / Exhaust Gas Anomaly":
        st.session_state['input_Cylinder1_Exhaust_Temp'] = 595.0
        st.session_state['input_Cylinder2_Exhaust_Temp'] = 585.0
        st.session_state['input_Cylinder3_Exhaust_Temp'] = 580.0
        st.session_state['input_Cylinder4_Exhaust_Temp'] = 605.0
    elif scenario_name == "Radial Engine Vibration Fault":
        st.session_state['input_Vibration_X'] = 0.47
        st.session_state['input_Vibration_Y'] = 0.44
    elif scenario_name == "Lubrication System Thermal Anomaly":
        st.session_state['input_Oil_Temp'] = 114.0
        st.session_state['input_Oil_Pressure'] = 0.7
    elif scenario_name == "Air Intake Pressure / Turbocharger Fault":
        st.session_state['input_Air_Pressure'] = 0.42
        st.session_state['input_Engine_Load'] = 98.0
    elif scenario_name == "Lubrication Pressure & Axial Vibration Fault":
        st.session_state['input_Oil_Pressure'] = 0.55
        st.session_state['input_Vibration_Z'] = 0.54

    run_prediction_pipeline(log_history=True)

# Function to render visual "Choose a Model" UI component featuring EXACTLY the 4 algorithms
def render_choose_a_model_ui():
    current_selected = st.session_state.get('active_model_key', 'decision_tree')
    
    render_html("""
    <div class="choose-model-card-box" style="background-color: #121110; border: 1px solid #28231e; border-radius: 18px; padding: 1.5rem; margin-bottom: 1.2rem; box-shadow: 0 12px 30px -5px rgba(0, 0, 0, 0.6);">
        <div style="display: flex; align-items: center; gap: 0.9rem;">
            <div style="background-color: #271f16; border: 1px solid #3d2f20; border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">🧠</div>
            <div>
                <h3 style="color: #f9fafb; font-size: 1.35rem; font-weight: 800; margin: 0; line-height: 1.2;">Choose a Model</h3>
                <p style="color: #9ca3af; font-size: 0.88rem; margin: 0.2rem 0 0 0; font-weight: 400;">Pick one machine learning algorithm to perform engine fault prediction</p>
            </div>
        </div>
    </div>
    """)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    models_info = [
        {"key": "decision_tree", "name": "Decision Tree", "badge": "TREE-BASED", "icon": "🌳", "badge_bg": "#064e3b", "badge_color": "#34d399", "col": col_m1},
        {"key": "svm", "name": "SVM", "badge": "KERNEL", "icon": "📐", "badge_bg": "#78350f", "badge_color": "#fbbf24", "col": col_m2},
        {"key": "random_forest", "name": "Random Forest", "badge": "ENSEMBLE", "icon": "🌲", "badge_bg": "#065f46", "badge_color": "#38bdf8", "col": col_m3},
        {"key": "xgboost", "name": "XGBoost", "badge": "BOOSTING", "icon": "🚀", "badge_bg": "#4c1d95", "badge_color": "#c084fc", "col": col_m4}
    ]
    
    for m in models_info:
        is_selected = (current_selected == m['key'])
        
        border_style = "2px solid #38bdf8" if is_selected else "1px solid #2b251f"
        bg_style = "radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%)" if is_selected else "#161412"
        shadow_style = "0 0 20px rgba(56, 189, 248, 0.35)" if is_selected else "0 4px 10px rgba(0,0,0,0.3)"
        check_mark = "✔" if is_selected else "◯"
        check_color = "#38bdf8" if is_selected else "#4b5563"
        
        with m['col']:
            render_html(f"""
            <div style="background: {bg_style}; border: {border_style}; box-shadow: {shadow_style}; border-radius: 14px; padding: 1.1rem 0.6rem; text-align: center; position: relative; transition: all 0.2s ease;">
                <div style="position: absolute; top: 8px; right: 10px; color: {check_color}; font-size: 0.95rem; font-weight: bold;">{check_mark}</div>
                <div style="font-size: 2.0rem; margin-bottom: 0.2rem;">{m['icon']}</div>
                <div style="color: #f9fafb; font-weight: 750; font-size: 0.98rem; margin-bottom: 0.4rem;">{m['name']}</div>
                <span style="background-color: {m['badge_bg']}; color: {m['badge_color']}; font-size: 0.70rem; font-weight: 800; padding: 0.2rem 0.60rem; border-radius: 10px; letter-spacing: 0.05em;">{m['badge']}</span>
            </div>
            """)
            st.write("")
            
            button_text = f"✔ Selected" if is_selected else f"Use {m['name']}"
            if st.button(button_text, key=f"btn_card_{m['key']}", use_container_width=True):
                st.session_state['active_model_key'] = m['key']
                st.session_state['active_model_name'] = MODEL_NAME_MAP.get(m['key'], m['name'])
                run_prediction_pipeline(log_history=True)
                st.rerun()



# Sidebar Navigation Panel
with st.sidebar:
    render_html("""
    <div style="padding: 0.2rem 0 0.8rem 0;">
        <div style="font-size: 1.25rem; font-weight: 800; color: #38bdf8; letter-spacing: 0.02em;">
            ⚓ NAUTILUS OS
        </div>
        <div style="font-size: 0.74rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px;">
            Marine Diagnostics Console
        </div>
    </div>
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 0.45rem 0.75rem; margin-bottom: 1.0rem; font-size: 0.76rem; color: #34d399; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
        <span style="height: 8px; width: 8px; background-color: #10b981; border-radius: 50%; display: inline-block; box-shadow: 0 0 8px #10b981;"></span>
        TELEMETRY LINK: ONLINE
    </div>
    """)

page_selection = st.sidebar.radio(
    "Navigation Console",
    [
        "🏠 Dashboard",
        "🔍 Prediction",
        "📊 Model Performance",
        "ℹ About Project"
    ]
)

current_key = st.session_state.get('active_model_key', 'decision_tree')
current_disp_name = MODEL_NAME_MAP.get(current_key, "Decision Tree")
current_index = MODEL_OPTIONS.index(current_disp_name) if current_disp_name in MODEL_OPTIONS else 0

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Selection")
active_model_disp = st.sidebar.selectbox(
    "Active Classifier Algorithm:",
    MODEL_OPTIONS,
    index=current_index,
    key="sidebar_active_model_selectbox"
)

selected_key_from_sidebar = MODEL_KEY_MAP[active_model_disp]
if selected_key_from_sidebar != st.session_state['active_model_key']:
    st.session_state['active_model_key'] = selected_key_from_sidebar
    st.session_state['active_model_name'] = active_model_disp
    run_prediction_pipeline(log_history=True)
    st.rerun()

# Dynamic updates for current selection
pred_label, confidence_score, health_score, pred_probs = run_prediction_pipeline()

# Render Pages
if page_selection == "🏠 Dashboard":
    # ------------------ HOME PAGE ------------------
    st.markdown("<div class='main-title'>⚓ Marine Engine Predictive Maintenance System</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>Real-Time Sensor Telemetry & Multi-Model Machine Learning Fault Diagnostics</div>", unsafe_allow_html=True)
    
    fault_info = FAULT_CLASSES[pred_label]
    status_color = fault_info['color']
    severity_label = fault_info['severity']
    status_icon = "🟢" if severity_label == "Healthy" else ("🟠" if severity_label == "Warning" else "🔴")
    active_m_disp = st.session_state.get('active_model_name', 'Decision Tree')
    
    # 4-Column Compact Metric Banner
    col_mb1, col_mb2, col_mb3, col_mb4 = st.columns(4)
    
    with col_mb1:
        render_html(f"""
        <div class="white-card" style="border-top: 4px solid {status_color} !important; padding: 1.1rem !important; margin-bottom: 0.5rem !important;">
            <h3>Engine Status</h3>
            <p class="card-value" style="color: {status_color} !important; font-size: 1.35rem !important;">{status_icon} {severity_label}</p>
            <p class="card-desc">Overall Severity Index</p>
        </div>
        """)
        
    with col_mb2:
        render_html(f"""
        <div class="white-card" style="border-top: 4px solid {status_color} !important; padding: 1.1rem !important; margin-bottom: 0.5rem !important;">
            <h3>Predicted Fault</h3>
            <p class="card-value" style="font-size: 1.1rem !important; line-height: 1.3; color: #f8fafc !important;">{fault_info['name']}</p>
            <p class="card-desc">Target Classifier Diagnosis</p>
        </div>
        """)
        
    with col_mb3:
        render_html(f"""
        <div class="white-card" style="border-top: 4px solid {status_color} !important; padding: 1.1rem !important; margin-bottom: 0.5rem !important;">
            <h3>Engine Health</h3>
            <p class="card-value" style="color: {status_color} !important; font-size: 1.35rem !important;">{health_score:.1f}%</p>
            <p class="card-desc">Physical Health Index</p>
        </div>
        """)
        
    with col_mb4:
        render_html(f"""
        <div class="white-card" style="border-top: 4px solid #38bdf8 !important; padding: 1.1rem !important; margin-bottom: 0.5rem !important;">
            <h3>Active Classifier</h3>
            <p class="card-value" style="color: #38bdf8 !important; font-size: 1.15rem !important; line-height: 1.3;">{active_m_disp}</p>
            <p class="card-desc">Selected ML Algorithm</p>
        </div>
        """)

    # Interactive Real-Time Telemetry Monitor Summary Card Grid
    st.markdown("<div class='section-header' style='margin-top: 1.0rem;'>📡 Live Telemetry Sensor Status</div>", unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    sens_summary = [
        ("Shaft RPM", st.session_state['input_Shaft_RPM'], "RPM", "Shaft_RPM", col_s1),
        ("Fuel Flow", st.session_state['input_Fuel_Flow'], "L/h", "Fuel_Flow", col_s2),
        ("Oil Temp", st.session_state['input_Oil_Temp'], "°C", "Oil_Temp", col_s3),
        ("Oil Pressure", st.session_state['input_Oil_Pressure'], "bar", "Oil_Pressure", col_s4),
        ("Vibration X", st.session_state['input_Vibration_X'], "g", "Vibration_X", col_s5)
    ]
    
    for name, val, unit, key, col in sens_summary:
        st_label, st_col, st_bg = check_sensor_status(key, val)
        with col:
            render_html(f"""
            <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 0.9rem 0.5rem; text-align: center;">
                <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">{name}</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff; margin: 0.2rem 0;">{val} <span style="font-size: 0.70rem; color: #64748b;">{unit}</span></div>
                <span class="status-pill" style="background-color: {st_bg}; color: {st_col};">{st_label}</span>
            </div>
            """)

    # Interactive Live Cylinder Combustion & Exhaust Monitor
    st.markdown("<div class='section-header' style='margin-top: 1.2rem;'>📊 Live Cylinder Combustion & Exhaust Monitor</div>", unsafe_allow_html=True)
    col_dash_c1, col_dash_c2 = st.columns(2)
    
    with col_dash_c1:
        cyl_pressures = [
            st.session_state['input_Cylinder1_Pressure'],
            st.session_state['input_Cylinder2_Pressure'],
            st.session_state['input_Cylinder3_Pressure'],
            st.session_state['input_Cylinder4_Pressure']
        ]
        df_cp = pd.DataFrame({
            'Cylinder': ['Cyl 1', 'Cyl 2', 'Cyl 3', 'Cyl 4'],
            'Pressure (bar)': cyl_pressures
        })
        fig_cp = px.bar(
            df_cp, x='Cylinder', y='Pressure (bar)',
            title='Combustion Pressures across Cylinders 1-4',
            text_auto='.1f',
            color='Pressure (bar)',
            color_continuous_scale='Tealgrn'
        )
        fig_cp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff', margin=dict(l=20, r=20, t=35, b=20), height=260,
            coloraxis_showscale=False
        )
        fig_cp.update_traces(textposition='outside', textfont=dict(color='#ffffff', size=11))
        st.plotly_chart(fig_cp, use_container_width=True)

    with col_dash_c2:
        cyl_temps = [
            st.session_state['input_Cylinder1_Exhaust_Temp'],
            st.session_state['input_Cylinder2_Exhaust_Temp'],
            st.session_state['input_Cylinder3_Exhaust_Temp'],
            st.session_state['input_Cylinder4_Exhaust_Temp']
        ]
        df_ct = pd.DataFrame({
            'Cylinder': ['Cyl 1', 'Cyl 2', 'Cyl 3', 'Cyl 4'],
            'Exhaust Temp (°C)': cyl_temps
        })
        fig_ct = px.bar(
            df_ct, x='Cylinder', y='Exhaust Temp (°C)',
            title='Exhaust Gas Temperatures across Cylinders 1-4',
            text_auto='.1f',
            color='Exhaust Temp (°C)',
            color_continuous_scale='Oranges'
        )
        fig_ct.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff', margin=dict(l=20, r=20, t=35, b=20), height=260,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_ct, use_container_width=True)

elif page_selection == "🔍 Prediction":
    # ------------------ PREDICTION PAGE ------------------
    st.markdown("<div class='main-title'>🔍 Predictive Diagnostics Form</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>Adjust engine signals or load a preset scenario to perform maintenance diagnostics.</div>", unsafe_allow_html=True)
    
    # Render Visual Model Picker Cards Grid (EXACTLY 4 models)
    render_choose_a_model_ui()
    
    # Preset Selector and Reset Control
    col_p1, col_p2 = st.columns([3, 1])
    
    with col_p1:
        preset_choice = st.selectbox(
            "⚡ Load System Anomaly Presets:",
            [
                "Normal Operation",
                "Fuel Delivery System Anomaly",
                "Low Cylinder Compression Pressure",
                "Combustion Heat / Exhaust Gas Anomaly",
                "Radial Engine Vibration Fault",
                "Lubrication System Thermal Anomaly",
                "Air Intake Pressure / Turbocharger Fault",
                "Lubrication Pressure & Axial Vibration Fault"
            ]
        )
    with col_p2:
        st.write("")
        st.write("")
        if st.button("Load Preset", use_container_width=True):
            load_preset(preset_choice)
            st.success(f"Loaded telemetry values for: {preset_choice}")
            st.rerun()
            
    st.markdown("<div class='section-header'>🎛️ Physical Sensor Measurements</div>", unsafe_allow_html=True)
    
    # 3-column input field form structure with live status badges
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("##### ⚙️ Propulsion States")
        
        st.slider("Shaft RPM (rotations/min)", 750.0, 1150.0, key="input_Shaft_RPM", step=1.0)
        st_lbl, st_color, st_bg = check_sensor_status('Shaft_RPM', st.session_state['input_Shaft_RPM'])
        render_html(f"<span class='status-pill' style='background:{st_bg}; color:{st_color}; margin-bottom: 15px;'>RPM Status: {st_lbl}</span>")
        
        st.slider("Engine Load (%)", 25.0, 110.0, key="input_Engine_Load", step=0.5)
        st_lbl, st_color, st_bg = check_sensor_status('Engine_Load', st.session_state['input_Engine_Load'])
        render_html(f"<span class='status-pill' style='background:{st_bg}; color:{st_color}; margin-bottom: 15px;'>Load Status: {st_lbl}</span>")
        
        st.slider("Fuel Flow (L/h)", 60.0, 190.0, key="input_Fuel_Flow", step=1.0)
        st_lbl, st_color, st_bg = check_sensor_status('Fuel_Flow', st.session_state['input_Fuel_Flow'])
        render_html(f"<span class='status-pill' style='background:{st_bg}; color:{st_color}; margin-bottom: 15px;'>Fuel Status: {st_lbl}</span>")
        
        st.slider("Air Pressure (bar)", 0.3, 1.6, key="input_Air_Pressure", step=0.01)
        st_lbl, st_color, st_bg = check_sensor_status('Air_Pressure', st.session_state['input_Air_Pressure'])
        render_html(f"<span class='status-pill' style='background:{st_bg}; color:{st_color}; margin-bottom: 15px;'>Boost Status: {st_lbl}</span>")
        
        st.slider("Ambient Temperature (°C)", 15.0, 40.0, key="input_Ambient_Temp", step=0.5)
        st.slider("Oil Temperature (°C)", 60.0, 115.0, key="input_Oil_Temp", step=0.5)
        st_lbl, st_color, st_bg = check_sensor_status('Oil_Temp', st.session_state['input_Oil_Temp'])
        render_html(f"<span class='status-pill' style='background:{st_bg}; color:{st_color}; margin-bottom: 15px;'>Oil Temp Status: {st_lbl}</span>")
        
    with col_f2:
        st.markdown("##### 📳 Vibration Channels & Oil")
        st.slider("Oil Pressure (bar)", 0.4, 5.2, key="input_Oil_Pressure", step=0.05)
        st_lbl, st_color, st_bg = check_sensor_status('Oil_Pressure', st.session_state['input_Oil_Pressure'])
        render_html(f"<span class='status-pill' style='background:{st_bg}; color:{st_color}; margin-bottom: 15px;'>Oil Pressure Status: {st_lbl}</span>")
        
        st.slider("Vibration X (g)", 0.0, 0.5, key="input_Vibration_X", step=0.001)
        st.slider("Vibration Y (g)", 0.0, 0.5, key="input_Vibration_Y", step=0.001)
        st.slider("Vibration Z (g)", 0.0, 0.6, key="input_Vibration_Z", step=0.001)
        st_lbl, st_color, st_bg = check_sensor_status('Vibration_Z', st.session_state['input_Vibration_Z'])
        render_html(f"<span class='status-pill' style='background:{st_bg}; color:{st_color}; margin-bottom: 15px;'>Axial Vibration Status: {st_lbl}</span>")
        
        st.markdown("##### 🛢️ Combustion Pressures (bar)")
        st.slider("Cylinder 1 Pressure", 85.0, 190.0, key="input_Cylinder1_Pressure", step=1.0)
        st.slider("Cylinder 2 Pressure", 90.0, 190.0, key="input_Cylinder2_Pressure", step=1.0)
        
    with col_f3:
        st.markdown("##### 🛢️ Combustion Pressures & Exhausts")
        st.slider("Cylinder 3 Pressure (bar)", 85.0, 190.0, key="input_Cylinder3_Pressure", step=1.0)
        st.slider("Cylinder 4 Pressure (bar)", 85.0, 190.0, key="input_Cylinder4_Pressure", step=1.0)
        st_lbl, st_color, st_bg = check_sensor_status('Cylinder4_Pressure', st.session_state['input_Cylinder4_Pressure'])
        render_html(f"<span class='status-pill' style='background:{st_bg}; color:{st_color}; margin-bottom: 15px;'>Cylinder Pressure Status: {st_lbl}</span>")
        
        st.markdown("##### 🔥 Exhaust Temperatures (°C)")
        st.slider("Cylinder 1 Exhaust Temperature", 290.0, 620.0, key="input_Cylinder1_Exhaust_Temp", step=1.0)
        st.slider("Cylinder 2 Exhaust Temperature", 310.0, 600.0, key="input_Cylinder2_Exhaust_Temp", step=1.0)
        st.slider("Cylinder 3 Exhaust Temperature", 300.0, 610.0, key="input_Cylinder3_Exhaust_Temp", step=1.0)
        st.slider("Cylinder 4 Exhaust Temperature", 310.0, 620.0, key="input_Cylinder4_Exhaust_Temp", step=1.0)
        st_lbl, st_color, st_bg = check_sensor_status('Cylinder4_Exhaust_Temp', st.session_state['input_Cylinder4_Exhaust_Temp'])
        render_html(f"<span class='status-pill' style='background:{st_bg}; color:{st_color}; margin-bottom: 15px;'>Exhaust Temp Status: {st_lbl}</span>")

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        if st.button("🚀 PREDICT ENGINE FAULT", use_container_width=True):
            run_prediction_pipeline(log_history=True)
            st.success(f"Diagnostic prediction executed using trained {st.session_state.get('active_model_name', 'model')}!")
            st.rerun()
    with col_btn2:
        if st.button("🔄 Reset Baseline", use_container_width=True):
            for k, v in DEFAULT_SENSOR_VALUES.items():
                st.session_state[f"input_{k}"] = v
            run_prediction_pipeline(log_history=True)
            st.info("Reset telemetry values to nominal engine baseline.")
            st.rerun()

    # Live prediction results rendering
    fault_info = FAULT_CLASSES[pred_label]
    color = fault_info['color']
    severity = fault_info['severity']
        
    if severity == "Healthy":
        card_bg = "rgba(16, 185, 129, 0.08)"
        card_border = "#10b981"
        card_text = "#6ee7b7"
    elif severity == "Warning":
        card_bg = "rgba(245, 158, 11, 0.08)"
        card_border = "#f59e0b"
        card_text = "#fde047"
    else:
        card_bg = "rgba(239, 68, 68, 0.08)"
        card_border = "#ef4444"
        card_text = "#fca5a5"
        
    st.markdown("<div class='section-header'>📊 Live Classification Result Metrics</div>", unsafe_allow_html=True)
    
    render_html(f"""
    <div style="display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;">
        <div class="white-card" style="flex: 1; min-width: 250px; border-top: 4px solid {color} !important;">
            <h3>Predicted Fault</h3>
            <p class="card-value" style="color: {color} !important; font-size: 1.25rem !important; line-height: 1.4;">{fault_info['name']}</p>
            <p class="card-desc">Output Target Diagnostics</p>
        </div>
        <div class="white-card" style="flex: 1; min-width: 250px; border-top: 4px solid {color} !important;">
            <h3>Prediction Confidence ({st.session_state.get('active_model_name', 'Active Model')})</h3>
            <p class="card-value">{confidence_score:.2%}</p>
            <p class="card-desc">Model Classification probability</p>
        </div>
        <div class="white-card" style="flex: 1; min-width: 250px; border-top: 4px solid {color} !important;">
            <h3>Engine Health</h3>
            <p class="card-value" style="color: {color} !important;">{health_score:.1f}%</p>
            <p class="card-desc">Propulsion Plant Health Index</p>
        </div>
    </div>
    """)

    # Interactive Visualizers: Fault Probabilities & Health Gauge
    col_vis1, col_vis2 = st.columns([6, 4])
    
    with col_vis1:
        class_names = [FAULT_CLASSES[i]['name'] for i in range(8)]
        df_probs = pd.DataFrame({
            'Fault Diagnosis Class': class_names,
            'Probability': pred_probs
        })
        
        bar_colors = ['#10b981' if i == 0 else ('#ef4444' if i == pred_label else '#334155') for i in range(8)]
        
        fig_prob = px.bar(
            df_probs, y='Fault Diagnosis Class', x='Probability', orientation='h',
            title=f'🎯 {st.session_state.get("active_model_name", "Active Model")} Fault Class Probabilities',
            text_auto='.1%',
            labels={'Probability': 'Probability', 'Fault Diagnosis Class': ''}
        )
        fig_prob.update_traces(marker_color=bar_colors, textposition='outside', textfont=dict(color='#f8fafc', size=11, family='Inter'))
        fig_prob.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc', family='Inter'),
            xaxis=dict(range=[0, 1.18], gridcolor='rgba(255,255,255,0.08)', tickfont=dict(color='#94a3b8')),
            yaxis=dict(autorange="reversed", tickfont=dict(color='#f8fafc', size=11)),
            margin=dict(l=10, r=20, t=40, b=20), height=340
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        
    with col_vis2:
        gauge_color = "#10b981" if health_score >= 80 else ("#f59e0b" if health_score >= 50 else "#ef4444")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            number = {'suffix': "%", 'font': {'color': gauge_color, 'size': 36, 'family': 'Inter'}},
            domain = {'x': [0, 1]},
            title = {'text': "⚓ Engine Health Index Gauge", 'font': {'size': 15, 'color': '#f8fafc', 'family': 'Inter'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color': '#94a3b8'}},
                'bar': {'color': gauge_color},
                'bgcolor': "rgba(15, 23, 42, 0.6)",
                'borderwidth': 1,
                'bordercolor': "rgba(255, 255, 255, 0.1)",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.15)'},
                    {'range': [50, 80], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [80, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f8fafc', family='Inter'),
            margin=dict(l=20, r=20, t=50, b=20), height=340
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Dynamic Maintenance Directives & Recommendation Card
    render_html(f"""
    <div class="white-card" style="border-left: 5px solid {card_border} !important; background: {card_bg} !important; border: 1px solid rgba(255,255,255,0.08) !important; padding: 1.5rem !important; margin-top: 1rem;">
        <h3 style="color: {card_text} !important; font-size: 0.95rem !important;">🛠️ Maintenance Directives & Recommendation</h3>
        <p style="font-size: 1.05rem !important; font-weight: 500 !important; color: #f8fafc !important; margin-top: 0.6rem; margin-bottom: 0; line-height: 1.6;">
            {fault_info['rec']}
        </p>
    </div>
    """)

elif page_selection == "📊 Model Performance":
    # ------------------ MODEL PERFORMANCE PAGE ------------------
    st.markdown("<div class='main-title'>📊 Classifiers Benchmarking & Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>Comparison matrix and diagnostic metrics for the four machine learning models trained on marine telemetry.</div>", unsafe_allow_html=True)
    
    if metrics_payload and 'metrics' in metrics_payload:
        m_dict = metrics_payload['metrics']
        current_active = st.session_state.get('active_model_key', 'decision_tree')
        
        def extract_metrics(m_key):
            if m_key in m_dict:
                v = m_dict[m_key]
                return {
                    'accuracy': v['accuracy'],
                    'precision': v['report']['macro avg']['precision'],
                    'recall': v['report']['macro avg']['recall'],
                    'f1': v['report']['macro avg']['f1-score']
                }
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
        dt = extract_metrics('decision_tree')
        svm = extract_metrics('svm')
        rf = extract_metrics('random_forest')
        xgb = extract_metrics('xgboost')
        
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        
        # 1. Decision Tree
        with col_c1:
            is_act = (current_active == 'decision_tree')
            border_c = "#34d399" if is_act else "#1f2937"
            bg_c = "rgba(52, 211, 153, 0.08)" if is_act else "#111827"
            shadow_c = "0 0 18px rgba(52, 211, 153, 0.35)" if is_act else "none"
            badge_html = "<span style='position: absolute; top: -12px; right: 12px; background-color: #34d399; color: #000000; font-size: 0.70rem; font-weight: 800; padding: 3px 8px; border-radius: 10px;'>ACTIVE MODEL</span>" if is_act else "<span style='position: absolute; top: -12px; right: 12px; background-color: #064e3b; color: #34d399; font-size: 0.70rem; font-weight: 800; padding: 3px 8px; border-radius: 10px;'>TREE MODEL</span>"
            
            c_html = f"""<div style="background-color: {bg_c}; border: 2px solid {border_c}; box-shadow: {shadow_c}; border-radius: 14px; padding: 1.4rem; position: relative;">{badge_html}<h3 style="color: #34d399 !important; font-size: 1.15rem; margin-top: 0.4rem; font-weight: 800;">Decision Tree</h3><hr style="margin: 0.6rem 0; border: 0; border-top: 1px solid #374151;"><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Accuracy:</b> <span style="color: #38bdf8; font-weight: 800;">{dt['accuracy']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Precision:</b> <span style="color: #f3f4f6; font-weight: 700;">{dt['precision']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Recall:</b> <span style="color: #f3f4f6; font-weight: 700;">{dt['recall']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>F1 Score:</b> <span style="color: #f3f4f6; font-weight: 700;">{dt['f1']:.2%}</span></p></div>"""
            st.markdown(c_html, unsafe_allow_html=True)
            st.write("")
            btn_txt = "✔ Active" if is_act else "Activate Decision Tree"
            if st.button(btn_txt, key="perf_btn_dt", use_container_width=True):
                st.session_state['active_model_key'] = 'decision_tree'
                st.session_state['active_model_name'] = 'Decision Tree'
                run_prediction_pipeline(log_history=True)
                st.rerun()
            
        # 2. Support Vector Machine (SVM)
        with col_c2:
            is_act = (current_active == 'svm')
            border_c = "#ef4444" if is_act else "#374151"
            bg_c = "rgba(239, 68, 68, 0.08)" if is_act else "#111827"
            shadow_c = "0 0 18px rgba(239, 68, 68, 0.35)" if is_act else "none"
            badge_html = "<span style='position: absolute; top: -12px; right: 12px; background-color: #ef4444; color: #ffffff; font-size: 0.70rem; font-weight: 800; padding: 3px 8px; border-radius: 10px;'>⭐ HIGHEST ACCURACY</span>"
            
            c_html = f"""<div style="background-color: {bg_c}; border: 2px solid {border_c}; box-shadow: {shadow_c}; border-radius: 14px; padding: 1.4rem; position: relative;">{badge_html}<h3 style="color: #f87171 !important; font-size: 1.15rem; margin-top: 0.4rem; font-weight: 800;">SVM (RBF Kernel)</h3><hr style="margin: 0.6rem 0; border: 0; border-top: 1px solid #374151;"><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Accuracy:</b> <span style="color: #38bdf8; font-weight: 800;">{svm['accuracy']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Precision:</b> <span style="color: #f3f4f6; font-weight: 700;">{svm['precision']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Recall:</b> <span style="color: #f3f4f6; font-weight: 700;">{svm['recall']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>F1 Score:</b> <span style="color: #f3f4f6; font-weight: 700;">{svm['f1']:.2%}</span></p></div>"""
            st.markdown(c_html, unsafe_allow_html=True)
            st.write("")
            btn_txt = "✔ Active" if is_act else "Activate SVM"
            if st.button(btn_txt, key="perf_btn_svm", use_container_width=True):
                st.session_state['active_model_key'] = 'svm'
                st.session_state['active_model_name'] = 'Support Vector Machine (SVM)'
                run_prediction_pipeline(log_history=True)
                st.rerun()
            
        # 3. Random Forest
        with col_c3:
            is_act = (current_active == 'random_forest')
            border_c = "#f59e0b" if is_act else "#374151"
            bg_c = "rgba(245, 158, 11, 0.08)" if is_act else "#111827"
            shadow_c = "0 0 18px rgba(245, 158, 11, 0.4)" if is_act else "none"
            badge_html = "<span style='position: absolute; top: -12px; right: 12px; background-color: #f59e0b; color: #000000; font-size: 0.70rem; font-weight: 800; padding: 3px 8px; border-radius: 10px;'>⭐ TOP F1 SCORE</span>"
            
            c_html = f"""<div style="background-color: {bg_c}; border: 2px solid {border_c}; box-shadow: {shadow_c}; border-radius: 14px; padding: 1.4rem; position: relative;">{badge_html}<h3 style="color: #fbbf24 !important; font-size: 1.15rem; margin-top: 0.4rem; font-weight: 800;">Random Forest</h3><hr style="margin: 0.6rem 0; border: 0; border-top: 1px solid #374151;"><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Accuracy:</b> <span style="color: #38bdf8; font-weight: 800;">{rf['accuracy']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Precision:</b> <span style="color: #34d399; font-weight: 800;">{rf['precision']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Recall:</b> <span style="color: #34d399; font-weight: 800;">{rf['recall']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>F1 Score:</b> <span style="color: #fbbf24; font-weight: 800;">{rf['f1']:.2%}</span></p></div>"""
            st.markdown(c_html, unsafe_allow_html=True)
            st.write("")
            btn_txt = "✔ Active" if is_act else "Activate Random Forest"
            if st.button(btn_txt, key="perf_btn_rf", use_container_width=True):
                st.session_state['active_model_key'] = 'random_forest'
                st.session_state['active_model_name'] = 'Random Forest'
                run_prediction_pipeline(log_history=True)
                st.rerun()

        # 4. XGBoost
        with col_c4:
            is_act = (current_active == 'xgboost')
            border_c = "#c084fc" if is_act else "#374151"
            bg_c = "rgba(192, 132, 252, 0.08)" if is_act else "#111827"
            shadow_c = "0 0 18px rgba(192, 132, 252, 0.35)" if is_act else "none"
            badge_html = "<span style='position: absolute; top: -12px; right: 12px; background-color: #c084fc; color: #ffffff; font-size: 0.70rem; font-weight: 800; padding: 3px 8px; border-radius: 10px;'>BOOSTING</span>"
            
            c_html = f"""<div style="background-color: {bg_c}; border: 2px solid {border_c}; box-shadow: {shadow_c}; border-radius: 14px; padding: 1.4rem; position: relative;">{badge_html}<h3 style="color: #c084fc !important; font-size: 1.15rem; margin-top: 0.4rem; font-weight: 800;">XGBoost</h3><hr style="margin: 0.6rem 0; border: 0; border-top: 1px solid #374151;"><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Accuracy:</b> <span style="color: #38bdf8; font-weight: 800;">{xgb['accuracy']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Precision:</b> <span style="color: #f3f4f6; font-weight: 700;">{xgb['precision']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>Recall:</b> <span style="color: #f3f4f6; font-weight: 700;">{xgb['recall']:.2%}</span></p><p style="font-size: 0.95rem; color: #cbd5e1; margin: 0.4rem 0;"><b>F1 Score:</b> <span style="color: #f3f4f6; font-weight: 700;">{xgb['f1']:.2%}</span></p></div>"""
            st.markdown(c_html, unsafe_allow_html=True)
            st.write("")
            btn_txt = "✔ Active" if is_act else "Activate XGBoost"
            if st.button(btn_txt, key="perf_btn_xgb", use_container_width=True):
                st.session_state['active_model_key'] = 'xgboost'
                st.session_state['active_model_name'] = 'XGBoost'
                run_prediction_pipeline(log_history=True)
                st.rerun()

        st.markdown("<div class='section-header'>📊 Model Benchmark Charts</div>", unsafe_allow_html=True)
        col_ch1, col_ch2 = st.columns(2)
        
        with col_ch1:
            df_acc = pd.DataFrame({
                'Algorithm': ['SVM', 'Random Forest', 'XGBoost', 'Decision Tree'],
                'Accuracy': [svm['accuracy'], rf['accuracy'], xgb['accuracy'], dt['accuracy']]
            }).sort_values(by='Accuracy', ascending=False)
            
            fig_acc = px.bar(
                df_acc, x='Algorithm', y='Accuracy', color='Algorithm',
                color_discrete_map={
                    'Decision Tree': '#34d399',
                    'SVM': '#ef4444',
                    'Random Forest': '#fbbf24',
                    'XGBoost': '#c084fc'
                },
                title='Top 4 Model Accuracy Benchmarks',
                text_auto='.2%'
            )
            fig_acc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f8fafc', family='Inter'),
                xaxis=dict(tickfont=dict(color='#f8fafc', size=12)),
                yaxis=dict(range=[0.7, 1.0], gridcolor='rgba(255,255,255,0.15)', tickfont=dict(color='#f8fafc')),
                showlegend=False, margin=dict(l=30, r=20, t=40, b=30), height=320
            )
            fig_acc.update_traces(textposition='outside', textfont=dict(color='#ffffff', size=12, family='Inter'))
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_ch2:
            sel_imp_model = st.selectbox(
                "Select Model for Feature Importances:",
                ["Decision Tree", "Support Vector Machine (SVM)", "Random Forest", "XGBoost"]
            )
            imp_key = MODEL_KEY_MAP[sel_imp_model]
            imp_model_obj = models.get(imp_key)
            
            importances = None
            if imp_key == 'svm' and imp_model_obj is not None:
                if hasattr(imp_model_obj, 'support_vectors_') and hasattr(imp_model_obj, 'dual_coef_'):
                    weights = np.abs(imp_model_obj.dual_coef_).sum(axis=0)
                    importances = np.average(np.abs(imp_model_obj.support_vectors_), weights=weights, axis=0)
                    importances = importances / importances.sum()
            elif imp_model_obj and hasattr(imp_model_obj, 'feature_importances_'):
                importances = imp_model_obj.feature_importances_
                
            if importances is not None:
                df_imp = pd.DataFrame({
                    'Sensor Channel': [f.replace('_', ' ') for f in feature_cols],
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True).tail(8)
                
                chart_color = "#34d399" if imp_key == 'decision_tree' else ("#ef4444" if imp_key == 'svm' else ("#fbbf24" if imp_key == 'random_forest' else "#c084fc"))
                
                fig_imp = px.bar(
                    df_imp, x='Importance', y='Sensor Channel', orientation='h',
                    title=f'{sel_imp_model} Feature Importance (Top 8 Sensors)',
                    text_auto='.3f'
                )
                fig_imp.update_traces(marker_color=chart_color, textposition='outside', textfont=dict(color='#ffffff', size=11))
                fig_imp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    margin=dict(l=30, r=20, t=40, b=30), height=320
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("Feature importances not available for selected model.")

        col_ch3, col_ch4 = st.columns(2)
        
        with col_ch3:
            selected_cm = st.selectbox(
                "Select Model for Confusion Matrix Visualizer:",
                ["Decision Tree", "Support Vector Machine (SVM)", "Random Forest", "XGBoost"]
            )
            cm_mapping = {
                "Decision Tree": "decision_tree",
                "Support Vector Machine (SVM)": "svm",
                "Random Forest": "random_forest",
                "XGBoost": "xgboost"
            }
            cm_key = cm_mapping[selected_cm]
            if cm_key in m_dict:
                cm = np.array(m_dict[cm_key]['cm'])
                cm_short_labels = ["Normal", "Fuel Sys", "Compress", "Exhaust", "Vibration", "Lube Temp", "Turbo", "Lube/Axial"]
                fig_cm = px.imshow(
                    cm, text_auto=True, aspect="auto",
                    labels=dict(x="Predicted Diagnosis Class", y="True Diagnosis Class", color="Count"),
                    x=cm_short_labels,
                    y=cm_short_labels,
                    title=f"Confusion Matrix Heatmap: {selected_cm}",
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    xaxis=dict(tickangle=0, tickfont=dict(size=9, color='#e2e8f0')),
                    yaxis=dict(tickfont=dict(size=9, color='#e2e8f0')),
                    margin=dict(l=30, r=20, t=40, b=30), height=350
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_ch4:
            target_class_roc = st.selectbox(
                "Select Target Classification Class for ROC Curve:",
                options=list(FAULT_CLASSES.keys()),
                format_func=lambda x: FAULT_CLASSES[x]['name']
            )
            
            @st.cache_data
            def get_test_splits():
                csv_path = "marine_engine_fault_dataset (1).csv"
                if not os.path.exists(csv_path):
                    return None, None
                df = pd.read_csv(csv_path).dropna().drop_duplicates()
                x_data = df[feature_cols].values
                y_data = df['Fault_Label'].values
                
                from sklearn.model_selection import train_test_split
                _, x_test, _, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
                
                with open("scaler.pkl", "rb") as f_sc:
                    sc = pickle.load(f_sc)
                x_test_scaled = sc.transform(x_test)
                return y_test, x_test_scaled
                
            y_test, x_test_scaled = get_test_splits()
            
            if y_test is not None:
                fig_roc = go.Figure()
                line_colors = {
                    'decision_tree': '#34d399',
                    'svm': '#f43f5e',
                    'random_forest': '#fbbf24',
                    'xgboost': '#c084fc'
                }
                
                for m_k, m_v in models.items():
                    if hasattr(m_v, 'predict_proba'):
                        try:
                            probs = m_v.predict_proba(x_test_scaled)
                            y_test_bin = (y_test == target_class_roc).astype(int)
                            y_score = probs[:, target_class_roc]
                            
                            fpr, tpr, _ = roc_curve(y_test_bin, y_score)
                            roc_auc = auc(fpr, tpr)
                            
                            disp_name = MODEL_NAME_MAP.get(m_k, m_k)
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr, mode='lines',
                                name=f'{disp_name} (AUC={roc_auc:.3f})',
                                line=dict(color=line_colors.get(m_k, '#cccccc'), width=1.8)
                            ))
                        except Exception:
                            pass
                            
                fig_roc.add_shape(
                    type='line', line=dict(dash='dash', color='#475569', width=1.5),
                    x0=0, x1=1, y0=0, y1=1
                )
                fig_roc.update_layout(
                    title=f"ROC Curves: {FAULT_CLASSES[target_class_roc]['name']}",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[-0.02, 1.02]),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[-0.02, 1.02]),
                    legend=dict(
                        yanchor="bottom", y=0.04, xanchor="right", x=0.98,
                        bgcolor='rgba(15,23,42,0.85)', bordercolor='rgba(255,255,255,0.1)'
                    ),
                    margin=dict(l=30, r=20, t=40, b=30), height=350
                )
                st.plotly_chart(fig_roc, use_container_width=True)



elif page_selection == "ℹ About Project":
    # ------------------ ABOUT PROJECT PAGE ------------------
    st.markdown("<div class='main-title'>ℹ About the Project</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>Academic final-year project documentation and overview.</div>", unsafe_allow_html=True)
    
    col_ab1, col_ab2 = st.columns(2)
    
    with col_ab1:
        render_html("""
        <div class="white-card">
            <h3>🎯 Project Objective</h3>
            <p style="font-size: 0.95rem; line-height: 1.5; margin-top: 0.5rem; font-weight:500;">
                The primary objective of this project is to develop an intelligent, automated predictive maintenance system for marine vessel engines. 
                By continuously monitoring high-dimensional sensor telemetry, the system aims to identify premature mechanical degradation, valve/cylinder leakage, turbocharger blockages, and lubrication anomalies before they result in catastrophic failure. 
                This proactive maintenance paradigm minimizes vessel downtime, reduces fuel inefficiencies, and enhances crew safety during open-water operations.
            </p>
        </div>
        
        <div class="white-card">
            <h3>📋 Dataset Overview</h3>
            <p style="font-size: 0.95rem; line-height: 1.5; margin-top: 0.5rem; font-weight:500;">
                The underlying dataset is generated from physical simulation rigs and telemetry logs of a modern 4-cylinder medium-speed marine diesel propulsion engine. 
                It contains over 10,000 engine operational records across diverse loads and environmental configurations, capturing baseline signatures alongside induced degradation patterns for 7 core fault conditions.
            </p>
        </div>
        
        <div class="white-card">
            <h3>📡 18 Physical Sensor Features</h3>
            <div style="font-size: 0.88rem; line-height: 1.4; color: #cbd5e1; font-weight: 500; max-height: 180px; overflow-y: auto;">
                1. <b>Shaft RPM</b> (Propulsion core rotation speed)<br>
                2. <b>Engine Load</b> (Percentage engine output torque)<br>
                3. <b>Fuel Flow</b> (Fuel volumetric flow rate)<br>
                4. <b>Air Pressure</b> (Turbocharger boost air pressure)<br>
                5. <b>Ambient Temperature</b> (Machinery space temperature)<br>
                6. <b>Oil Temperature</b> (Lubricating system oil temperature)<br>
                7. <b>Oil Pressure</b> (Lubricating feed main pressure)<br>
                8-10. <b>Vibration X, Y, Z</b> (Radial/axial engine block displacement)<br>
                11-14. <b>Cylinder 1-4 Pressures</b> (Peak combustion cylinder pressures)<br>
                15-18. <b>Cylinder 1-4 Exhaust Temps</b> (Post-combustion gas temperatures)
            </div>
        </div>
        """)

    with col_ab2:
        render_html("""
        <div class="white-card">
            <h3>🤖 Exact 4 Machine Learning Algorithms Used</h3>
            <p style="font-size: 0.9rem; line-height: 1.5; margin-top: 0.5rem; color: #cbd5e1; font-weight: 500;">
                - <b>Decision Tree Classifier</b>: Interpretable tree-based hierarchical decision boundary separator.<br>
                - <b>Support Vector Machine (SVM)</b>: RBF-kernel high-dimensional distance separator achieving top classification accuracy.<br>
                - <b>Random Forest Classifier</b>: Ensemble bagging model utilizing decision tree ensembles for top F1 score.<br>
                - <b>XGBoost (Extreme Gradient Boosting)</b>: Gradient boosted decision trees optimized for fast inference.
            </p>
        </div>
        
        <div class="white-card">
            <h3>📈 Dynamic Prediction Process</h3>
            <ol style="font-size: 0.9rem; margin-top: 0.5rem; padding-left: 1.2rem; color: #cbd5e1; font-weight: 500; line-height: 1.6;">
                <li>Physical telemetry values are read from sensor controls or anomaly presets.</li>
                <li>Inputs are standardized using pre-fit <code>scaler.pkl</code> weights.</li>
                <li>Standardized feature vector is routed to the user's selected trained ML model.</li>
                <li>The target fault class and posterior probability vector are generated by the model.</li>
                <li>Engine Health Index, Fault Severity, and Maintenance Directives update dynamically.</li>
            </ol>
        </div>
        
        <div class="white-card">
            <h3>🎯 Expected Output</h3>
            <p style="font-size: 0.9rem; line-height: 1.5; margin-top: 0.5rem; color: #cbd5e1; font-weight: 500;">
                Output comprises three diagnostic indexes: <b>Engine Health Score</b> (0-100%), <b>Predicted Fault</b> diagnosis, <b>Model Confidence</b>, and the <b>Maintenance Crew Directives</b>.
            </p>
        </div>
        """)

# Footer
render_html("""
<div style='text-align:center; border-top:1px solid rgba(255,255,255,0.08); margin-top:3.5rem; padding-top:1.8rem; padding-bottom:1.8rem;'>
    <div style='font-size:0.82rem; color:#64748b; font-weight:600; letter-spacing:0.06em; text-transform:uppercase;'>
        ⚓ NAUTILUS TELEMETRY OS &bull; INDUSTRIAL MARINE ENGINE DIAGNOSTICS &bull; ENTERPRISE EDITION
    </div>
</div>
""")
