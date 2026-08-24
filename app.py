import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
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

# Fault Definitions
FAULT_CLASSES = {
    0: {
        "name": "Normal Operation",
        "desc": "All systems operating within baseline parameters.",
        "color": "#10b981", # Green
        "severity": "Healthy",
        "rec": "Continue nominal operations. Execute standard logbook entries. Scheduled maintenance in 250 hours."
    },
    1: {
        "name": "Fuel Delivery System Anomaly",
        "desc": "Fuel flow rate is disproportionately high relative to shaft RPM and load.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Isolate fuel delivery line. Inspect fuel primary filter elements. Test injectors 1-4 for spray anomalies."
    },
    2: {
        "name": "Low Cylinder Compression Pressure",
        "desc": "Pressure leakage detected across combustion chambers.",
        "color": "#f59e0b", # Orange
        "severity": "Warning",
        "rec": "Schedule static compression test. Inspect inlet/exhaust valve clearances. Audit crankcase pressure."
    },
    3: {
        "name": "Combustion Heat / Exhaust Gas Anomaly",
        "desc": "Exhaust gas temperature profile exceeds safety thresholds.",
        "color": "#f59e0b", # Orange
        "severity": "Warning",
        "rec": "Check cooling jacket water outlet temperature. Clean cooling path. Inspect exhaust manifold thermowells."
    },
    4: {
        "name": "Radial Engine Vibration Fault",
        "desc": "Excessive displacement recorded in the X and Y axes.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Halt engine if vibration exceeds safety RMS. Perform coupling laser alignment. Check engine mounts."
    },
    5: {
        "name": "Lubrication System Thermal Anomaly",
        "desc": "High sump oil temperature coupled with declining feed line pressure.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Verify oil cooler cooling water flow. Sample lube oil for viscosity check. Inspect regulation valves."
    },
    6: {
        "name": "Air Intake Pressure / Turbocharger Fault",
        "desc": "Boost air pressure drops significantly under engine load.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Inspect turbocharger rotor shaft for play. Inspect intake filter differential pressure."
    },
    7: {
        "name": "Lubrication Pressure & Axial Vibration Fault",
        "desc": "Severe drop in oil pressure accompanied by high vibration in the Z (axial) direction.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Shut down engine immediately. Conduct crankcase sump inspection for metal shavings. Audit thrust collar."
    }
}

# Custom Styling (Dark Navy Background with Clean Contrast White Cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;750;800&display=swap');

    /* Global setting */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0a1128 0%, #030712 100%) !important;
        color: #f3f4f6 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Adjust Streamlit padding to prevent header clipping */
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 2rem !important;
    }
    /* Header layout */
    .header-container {
        margin-bottom: 2rem;
    }
    .main-title {
        font-family: 'Inter', sans-serif !important;
        color: #38bdf8;
        font-size: 2.0rem;
        font-weight: 800;
        line-height: 1.3;
        margin-bottom: 0.2rem;
        word-wrap: break-word;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }

    /* White Cards style */
    .white-card {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border-radius: 12px !important;
        padding: 1.4rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
        border: 1px solid #e5e7eb !important;
        margin-bottom: 1rem !important;
        transition: transform 0.2s ease-in-out;
    }
    .white-card:hover {
        transform: translateY(-2px);
    }
    .white-card h3 {
        margin-top: 0 !important;
        color: #4b5563 !important;
        font-size: 0.9rem !important;
        font-weight: 750 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    .white-card p.card-value {
        color: #111827 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin: 0.4rem 0 0 0 !important;
    }
    .white-card p.card-desc {
        color: #6b7280 !important;
        font-size: 0.82rem !important;
        margin: 0.4rem 0 0 0 !important;
        font-weight: 500 !important;
    }

    /* Workflow layout */
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 2rem;
        margin-bottom: 2rem;
        width: 100%;
    }
    .workflow-step {
        background-color: #ffffff;
        color: #1f2937;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        text-align: center;
        width: 22%;
        min-height: 110px;
    }
    .workflow-step h4 {
        margin: 0.4rem 0;
        color: #111827;
        font-size: 1rem;
        font-weight: 750;
    }
    .workflow-step p {
        margin: 0;
        color: #6b7280;
        font-size: 0.8rem;
        line-height: 1.3;
    }
    .workflow-arrow {
        font-size: 1.8rem;
        color: #38bdf8;
        font-weight: bold;
    }

    /* Form styling and buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
    }

    /* Subsections and subheaders */
    .section-header {
        font-size: 1.4rem;
        font-weight: 750;
        color: #ffffff;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #38bdf8;
        padding-left: 0.6rem;
    }
    
    /* Preset selectors */
    .stSelectbox label, .stSlider label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scalar assets
@st.cache_resource
def load_ml_assets():
    model_files = {
        'logistic_regression': 'logistic_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'xgboost': 'xgboost_model.pkl',
        'decision_tree': 'decision_tree_model.pkl',
        'svm': 'svm_model.pkl',
        'knn': 'knn_model.pkl'
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

# Feature Column Definitions
feature_cols = [
    'Shaft_RPM', 'Engine_Load', 'Fuel_Flow', 'Air_Pressure', 'Ambient_Temp', 'Oil_Temp', 'Oil_Pressure',
    'Vibration_X', 'Vibration_Y', 'Vibration_Z', 'Cylinder1_Pressure', 'Cylinder1_Exhaust_Temp',
    'Cylinder2_Pressure', 'Cylinder2_Exhaust_Temp', 'Cylinder3_Pressure', 'Cylinder3_Exhaust_Temp',
    'Cylinder4_Pressure', 'Cylinder4_Exhaust_Temp'
]

# Initialize Session State values for 18 features
def init_session_state():
    defaults = {
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
    for k, v in defaults.items():
        if f"input_{k}" not in st.session_state:
            st.session_state[f"input_{k}"] = v
            
    if 'latest_pred' not in st.session_state:
        st.session_state['latest_pred'] = 0
    if 'latest_conf' not in st.session_state:
        st.session_state['latest_conf'] = 0.985
    if 'latest_health' not in st.session_state:
        st.session_state['latest_health'] = 98.2
    if 'predicted_clicked' not in st.session_state:
        st.session_state['predicted_clicked'] = False

init_session_state()

# Presets loading logic
def load_preset(scenario_name):
    st.session_state['predicted_clicked'] = False
    if scenario_name == "Normal Operation":
        st.session_state['input_Shaft_RPM'] = 960.0
        st.session_state['input_Engine_Load'] = 75.0
        st.session_state['input_Fuel_Flow'] = 130.0
        st.session_state['input_Air_Pressure'] = 1.15
        st.session_state['input_Ambient_Temp'] = 27.0
        st.session_state['input_Oil_Temp'] = 78.0
        st.session_state['input_Oil_Pressure'] = 3.4
        st.session_state['input_Vibration_X'] = 0.06
        st.session_state['input_Vibration_Y'] = 0.05
        st.session_state['input_Vibration_Z'] = 0.07
        st.session_state['input_Cylinder1_Pressure'] = 145.0
        st.session_state['input_Cylinder1_Exhaust_Temp'] = 420.0
        st.session_state['input_Cylinder2_Pressure'] = 145.0
        st.session_state['input_Cylinder2_Exhaust_Temp'] = 420.0
        st.session_state['input_Cylinder3_Pressure'] = 145.0
        st.session_state['input_Cylinder3_Exhaust_Temp'] = 420.0
        st.session_state['input_Cylinder4_Pressure'] = 145.0
        st.session_state['input_Cylinder4_Exhaust_Temp'] = 420.0
    elif scenario_name == "Fuel Delivery System Anomaly":
        st.session_state['input_Fuel_Flow'] = 188.0
        st.session_state['input_Engine_Load'] = 45.0
        st.session_state['input_Shaft_RPM'] = 820.0
    elif scenario_name == "Low Cylinder Compression Pressure":
        st.session_state['input_Cylinder1_Pressure'] = 90.0
        st.session_state['input_Cylinder2_Pressure'] = 93.0
        st.session_state['input_Cylinder3_Pressure'] = 89.0
        st.session_state['input_Cylinder4_Pressure'] = 91.0
    elif scenario_name == "Combustion Heat / Exhaust Gas Anomaly":
        st.session_state['input_Cylinder1_Exhaust_Temp'] = 590.0
        st.session_state['input_Cylinder2_Exhaust_Temp'] = 585.0
        st.session_state['input_Cylinder3_Exhaust_Temp'] = 580.0
        st.session_state['input_Cylinder4_Exhaust_Temp'] = 595.0
    elif scenario_name == "Radial Engine Vibration Fault":
        st.session_state['input_Vibration_X'] = 0.46
        st.session_state['input_Vibration_Y'] = 0.43
    elif scenario_name == "Lubrication System Thermal Anomaly":
        st.session_state['input_Oil_Temp'] = 114.0
        st.session_state['input_Oil_Pressure'] = 0.7
    elif scenario_name == "Air Intake Pressure / Turbocharger Fault":
        st.session_state['input_Air_Pressure'] = 0.42
        st.session_state['input_Engine_Load'] = 98.0
    elif scenario_name == "Lubrication Pressure & Axial Vibration Fault":
        st.session_state['input_Oil_Pressure'] = 0.55
        st.session_state['input_Vibration_Z'] = 0.54

# Sidebar Navigation Panel
st.sidebar.title("🚢 Marine Maintenance")
page_selection = st.sidebar.radio(
    "Navigation Console",
    [
        "🏠 Dashboard",
        "🔍 Prediction",
        "📊 Model Performance",
        "ℹ About Project"
    ]
)

# Render Pages
if page_selection == "🏠 Dashboard":
    # ------------------ HOME PAGE ------------------
    st.markdown("<div class='main-title'>⚓ Marine Engine Predictive Maintenance System</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>AI-Based Fault Detection and Engine Health Monitoring</div>", unsafe_allow_html=True)
    
    col_h1, col_h2 = st.columns([7, 3])
    
    with col_h1:
        
        # Latest states derived from st.session_state
        pred_label = st.session_state['latest_pred']
        fault_info = FAULT_CLASSES[pred_label]
        status_color = fault_info['color']
        severity_label = fault_info['severity']
        health_score = st.session_state['latest_health']
        
        # Indicator tag circles
        status_icon = "🟢" if severity_label == "Healthy" else ("🟠" if severity_label == "Warning" else "🔴")
        
        # Layout three information cards
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap;">
            <div class="white-card" style="flex: 1; min-width: 200px; border-top: 5px solid {status_color} !important;">
                <h3>Engine Status</h3>
                <p class="card-value" style="color: {status_color} !important;">{status_icon} {severity_label}</p>
                <p class="card-desc">Overall Severity Index</p>
            </div>
            <div class="white-card" style="flex: 1; min-width: 200px; border-top: 5px solid {status_color} !important;">
                <h3>Predicted Fault</h3>
                <p class="card-value" style="font-size: 1.15rem !important; line-height: 1.4; color: #111827;">{fault_info['name']}</p>
                <p class="card-desc">Target Classifier Diagnosis</p>
            </div>
            <div class="white-card" style="flex: 1; min-width: 200px; border-top: 5px solid {status_color} !important;">
                <h3>Engine Health</h3>
                <p class="card-value" style="color: {status_color} !important;">{health_score:.1f}%</p>
                <p class="card-desc">Physical Health Index</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_h2:
        # Display the engine illustration
        if os.path.exists("ship_engine.png"):
            st.image("ship_engine.png", use_container_width=True, caption="Ship Propulsion Engine Illustration")
        else:
            st.warning("Ship engine image asset not found.")

    # Simple workflow below
    st.markdown("<div class='section-header'>🔄 Predictive Maintenance Workflow</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-step">
            <div style="font-size: 1.8rem;">🔌</div>
            <h4>Sensor Data</h4>
            <p>18 real-time physical telemetry channels from cylinder valves, bearings, and sumps.</p>
        </div>
        <div class="workflow-arrow">➡</div>
        <div class="workflow-step">
            <div style="font-size: 1.8rem;">🧠</div>
            <h4>Machine Learning Model</h4>
            <p>Standardized feature scaling with Random Forest ensemble classifier prediction.</p>
        </div>
        <div class="workflow-arrow">➡</div>
        <div class="workflow-step">
            <div style="font-size: 1.8rem;">🔍</div>
            <h4>Fault Prediction</h4>
            <p>Detection of normal operations or 7 unique mechanical anomaly diagnoses.</p>
        </div>
        <div class="workflow-arrow">➡</div>
        <div class="workflow-step">
            <div style="font-size: 1.8rem;">🛠️</div>
            <h4>Maintenance Rec</h4>
            <p>Immediate action checklist and priority response crew directives.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page_selection == "🔍 Prediction":
    # ------------------ PREDICTION PAGE ------------------
    st.markdown("<div class='main-title'>🔍 Predictive Diagnostics Form</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>Adjust engine signals or load a preset scenario to perform maintenance diagnostics.</div>", unsafe_allow_html=True)
    
    # Preset Selector at top
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
    if st.button("Load Preset"):
        load_preset(preset_choice)
        st.success(f"Loaded variables for: {preset_choice}")
        
    st.markdown("<div class='section-header'>🎛️ Physical Sensor Measurements</div>", unsafe_allow_html=True)
    
    # 3-column input field form structure
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown("##### ⚙️ Propulsion States")
        st.slider("Shaft RPM (rotations/min)", 750.0, 1150.0, key="input_Shaft_RPM", step=1.0)
        st.slider("Engine Load (%)", 25.0, 110.0, key="input_Engine_Load", step=0.5)
        st.slider("Fuel Flow (L/h)", 60.0, 190.0, key="input_Fuel_Flow", step=1.0)
        st.slider("Air Pressure (bar)", 0.3, 1.6, key="input_Air_Pressure", step=0.01)
        st.slider("Ambient Temperature (°C)", 15.0, 40.0, key="input_Ambient_Temp", step=0.5)
        st.slider("Oil Temperature (°C)", 60.0, 115.0, key="input_Oil_Temp", step=0.5)
        
    with col_f2:
        st.markdown("##### 📳 Vibration Channels & Oil")
        st.slider("Oil Pressure (bar)", 0.4, 5.2, key="input_Oil_Pressure", step=0.05)
        st.slider("Vibration X (g)", 0.0, 0.5, key="input_Vibration_X", step=0.001)
        st.slider("Vibration Y (g)", 0.0, 0.5, key="input_Vibration_Y", step=0.001)
        st.slider("Vibration Z (g)", 0.0, 0.6, key="input_Vibration_Z", step=0.001)
        st.markdown("##### 🛢️ Combustion Pressures (bar)")
        st.slider("Cylinder 1 Pressure", 85.0, 190.0, key="input_Cylinder1_Pressure", step=1.0)
        st.slider("Cylinder 2 Pressure", 90.0, 190.0, key="input_Cylinder2_Pressure", step=1.0)
        
    with col_f3:
        st.markdown("##### 🛢️ Combustion Pressures & Exhausts")
        st.slider("Cylinder 3 Pressure (bar)", 85.0, 190.0, key="input_Cylinder3_Pressure", step=1.0)
        st.slider("Cylinder 4 Pressure (bar)", 85.0, 190.0, key="input_Cylinder4_Pressure", step=1.0)
        st.markdown("##### 🔥 Exhaust Temperatures (°C)")
        st.slider("Cylinder 1 Exhaust Temperature", 290.0, 620.0, key="input_Cylinder1_Exhaust_Temp", step=1.0)
        st.slider("Cylinder 2 Exhaust Temperature", 310.0, 600.0, key="input_Cylinder2_Exhaust_Temp", step=1.0)
        st.slider("Cylinder 3 Exhaust Temperature", 300.0, 610.0, key="input_Cylinder3_Exhaust_Temp", step=1.0)
        st.slider("Cylinder 4 Exhaust Temperature", 310.0, 620.0, key="input_Cylinder4_Exhaust_Temp", step=1.0)

    # Large Action Button
    st.write("")
    if st.button("🚀 Predict Engine Fault"):
        st.session_state['predicted_clicked'] = True
        
        # Load input values into vector
        input_vector = [st.session_state[f"input_{col}"] for col in feature_cols]
        scaled_vector = scaler.transform([input_vector])
        
        # Random forest classifier prediction
        model_obj = models['random_forest']
        pred_label = int(model_obj.predict(scaled_vector)[0])
        pred_probs = model_obj.predict_proba(scaled_vector)[0]
        confidence_score = float(pred_probs[pred_label])
        
        # Derive engine health score
        if pred_label == 0:
            health_score = 95.0 + (confidence_score * 4.8)
        else:
            # Anomaly health degradation
            health_score = max(5.0, 90.0 - (confidence_score * 45.0) - (np.random.random() * 8.0))
            
        st.session_state['latest_pred'] = pred_label
        st.session_state['latest_conf'] = confidence_score
        st.session_state['latest_health'] = health_score

    # Result Cards Display (After clicking Predict)
    if st.session_state['predicted_clicked']:
        pred_label = st.session_state['latest_pred']
        confidence_score = st.session_state['latest_conf']
        health_score = st.session_state['latest_health']
        
        fault_info = FAULT_CLASSES[pred_label]
        color = fault_info['color']
        severity = fault_info['severity']
        
        # Class styling properties
        if severity == "Healthy":
            card_bg = "#ecfdf5"
            card_text = "#065f46"
        elif severity == "Warning":
            card_bg = "#fffbeb"
            card_text = "#92400e"
        else:
            card_bg = "#fef2f2"
            card_text = "#991b1b"
            
        # Display Result Indicator cards
        st.markdown("<div class='section-header'>📊 Classification Result Metrics</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin-top: 1rem; flex-wrap: wrap;">
            <div class="white-card" style="flex: 1; min-width: 250px; border-top: 5px solid {color} !important;">
                <h3>Predicted Fault</h3>
                <p class="card-value" style="color: {color} !important; font-size: 1.25rem !important; line-height: 1.4;">{fault_info['name']}</p>
                <p class="card-desc">Output Target Diagnostics</p>
            </div>
            <div class="white-card" style="flex: 1; min-width: 250px; border-top: 5px solid {color} !important;">
                <h3>Prediction Confidence</h3>
                <p class="card-value">{confidence_score:.2%}</p>
                <p class="card-desc">Model Output Classification probability</p>
            </div>
            <div class="white-card" style="flex: 1; min-width: 250px; border-top: 5px solid {color} !important;">
                <h3>Engine Health</h3>
                <p class="card-value" style="color: {color} !important;">{health_score:.1f}%</p>
                <p class="card-desc">Propulsion Plant Health Index</p>
            </div>
        </div>
        
        <div class="white-card" style="border-left: 6px solid {color} !important; background-color: {card_bg} !important; color: {card_text} !important; padding: 1.5rem !important; margin-top: 1rem;">
            <h3 style="color: {card_text} !important; font-size: 0.95rem !important;">🛠️ Maintenance Recommendation</h3>
            <p style="font-size: 1.05rem !important; font-weight: 600 !important; color: {card_text} !important; margin-top: 0.6rem; margin-bottom: 0;">
                {fault_info['rec']}
            </p>
        </div>
        """, unsafe_allow_html=True)

elif page_selection == "📊 Model Performance":
    # ------------------ MODEL PERFORMANCE PAGE ------------------
    st.markdown("<div class='main-title'>📊 Classifiers Benchmarking & Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>Comparison matrix and diagnostic metrics for all six machine learning models trained.</div>", unsafe_allow_html=True)
    
    if metrics_payload and 'metrics' in metrics_payload:
        m_dict = metrics_payload['metrics']
        
        # Cards metrics payload extraction helper
        def extract_metrics(m_key):
            v = m_dict[m_key]
            return {
                'accuracy': v['accuracy'],
                'precision': v['report']['macro avg']['precision'],
                'recall': v['report']['macro avg']['recall'],
                'f1': v['report']['macro avg']['f1-score']
            }
            
        rf = extract_metrics('random_forest')
        xgb = extract_metrics('xgboost')
        dt = extract_metrics('decision_tree')
        lr = extract_metrics('logistic_regression')
        svm = extract_metrics('svm')
        knn = extract_metrics('knn')
        
        # Grid arrangement for model cards
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c4, col_c5, col_c6 = st.columns(3)
        
        # 1. Random Forest (Highlight best model)
        with col_c1:
            st.markdown(f"""
            <div class="white-card" style="border: 2px solid #fbbf24 !important; box-shadow: 0 0 15px rgba(251, 191, 36, 0.35) !important; position: relative;">
                <span style="position: absolute; top: -12px; right: 12px; background-color: #fbbf24; color: #000000; font-size: 0.72rem; font-weight: 800; padding: 2px 8px; border-radius: 10px; font-family: 'Inter', sans-serif;">⭐ BEST MODEL</span>
                <h3 style="color: #d97706 !important;">Random Forest</h3>
                <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Accuracy:</b> {rf['accuracy']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Precision:</b> {rf['precision']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Recall:</b> {rf['recall']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>F1 Score:</b> {rf['f1']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 2. XGBoost
        with col_c2:
            st.markdown(f"""
            <div class="white-card">
                <h3 style="color: #2563eb !important;">XGBoost</h3>
                <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Accuracy:</b> {xgb['accuracy']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Precision:</b> {xgb['precision']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Recall:</b> {xgb['recall']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>F1 Score:</b> {xgb['f1']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 3. Decision Tree
        with col_c3:
            st.markdown(f"""
            <div class="white-card">
                <h3 style="color: #4b5563 !important;">Decision Tree</h3>
                <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Accuracy:</b> {dt['accuracy']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Precision:</b> {dt['precision']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Recall:</b> {dt['recall']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>F1 Score:</b> {dt['f1']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 4. Logistic Regression
        with col_c4:
            st.markdown(f"""
            <div class="white-card">
                <h3 style="color: #4b5563 !important;">Logistic Regression</h3>
                <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Accuracy:</b> {lr['accuracy']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Precision:</b> {lr['precision']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Recall:</b> {lr['recall']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>F1 Score:</b> {lr['f1']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 5. SVM
        with col_c5:
            st.markdown(f"""
            <div class="white-card">
                <h3 style="color: #4b5563 !important;">SVM</h3>
                <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Accuracy:</b> {svm['accuracy']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Precision:</b> {svm['precision']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Recall:</b> {svm['recall']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>F1 Score:</b> {svm['f1']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # 6. KNN
        with col_c6:
            st.markdown(f"""
            <div class="white-card">
                <h3 style="color: #4b5563 !important;">KNN</h3>
                <hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e5e7eb;">
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Accuracy:</b> {knn['accuracy']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Precision:</b> {knn['precision']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>Recall:</b> {knn['recall']:.2%}</p>
                <p style="font-size: 0.95rem; color: #374151; margin: 0.25rem 0;"><b>F1 Score:</b> {knn['f1']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📊 Model Benchmark Charts</div>", unsafe_allow_html=True)
        col_ch1, col_ch2 = st.columns(2)
        
        with col_ch1:
            # 1. Accuracy Comparison Chart
            df_acc = pd.DataFrame({
                'Algorithm': ['Logistic Reg.', 'Decision Tree', 'KNN', 'Random Forest', 'XGBoost', 'SVM'],
                'Accuracy': [lr['accuracy'], dt['accuracy'], knn['accuracy'], rf['accuracy'], xgb['accuracy'], svm['accuracy']]
            }).sort_values(by='Accuracy', ascending=False)
            
            fig_acc = px.bar(
                df_acc, x='Algorithm', y='Accuracy', color='Accuracy',
                color_continuous_scale='Blues', title='Model Accuracy Benchmarks'
            )
            fig_acc.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff', yaxis=dict(range=[0.7, 1.0], gridcolor='rgba(255,255,255,0.05)'),
                coloraxis_showscale=False, margin=dict(l=30, r=20, t=40, b=30), height=300
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col_ch2:
            # 2. Feature Importance Chart (Random Forest)
            rf_model_obj = models.get('random_forest')
            if rf_model_obj and hasattr(rf_model_obj, 'feature_importances_'):
                importances = rf_model_obj.feature_importances_
                df_imp = pd.DataFrame({
                    'Sensor Channel': [f.replace('_', ' ') for f in feature_cols],
                    'Importance': importances
                }).sort_values(by='Importance', ascending=True).tail(8) # Top 8 features
                
                fig_imp = px.bar(
                    df_imp, x='Importance', y='Sensor Channel', orientation='h',
                    color='Importance', color_continuous_scale='Blues',
                    title='Random Forest Feature Importance (Top 8 Sensors)'
                )
                fig_imp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff', coloraxis_showscale=False,
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                    margin=dict(l=30, r=20, t=40, b=30), height=300
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("Feature importances not available.")

        col_ch3, col_ch4 = st.columns(2)
        
        with col_ch3:
            # 3. Confusion Matrix Chart
            selected_cm = st.selectbox(
                "Select Model for Confusion Matrix Visualizer:",
                ["Random Forest", "XGBoost", "Decision Tree", "Support Vector Machine", "K-Nearest Neighbors", "Logistic Regression"]
            )
            cm_mapping = {
                "Random Forest": "random_forest",
                "XGBoost": "xgboost",
                "Decision Tree": "decision_tree",
                "Support Vector Machine": "svm",
                "K-Nearest Neighbors": "knn",
                "Logistic Regression": "logistic_regression"
            }
            cm_key = cm_mapping[selected_cm]
            cm = np.array(m_dict[cm_key]['cm'])
            
            fig_cm = px.imshow(
                cm, text_auto=True, aspect="auto",
                labels=dict(x="Predicted Diagnosis Class", y="True Diagnosis Class", color="Count"),
                x=[FAULT_CLASSES[i]['name'] for i in range(8)],
                y=[FAULT_CLASSES[i]['name'] for i in range(8)],
                title=f"Confusion Matrix Heatmap: {selected_cm}",
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff', margin=dict(l=30, r=20, t=40, b=30), height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_ch4:
            # 4. ROC Curves (One-vs-Rest for target class)
            target_class_roc = st.selectbox(
                "Select Target Classification class for ROC Curve:",
                options=list(FAULT_CLASSES.keys()),
                format_func=lambda x: FAULT_CLASSES[x]['name']
            )
            
            # Helper to get test split
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
                    'logistic_regression': '#60a5fa',
                    'random_forest': '#10b981',
                    'xgboost': '#fbbf24',
                    'decision_tree': '#a78bfa',
                    'svm': '#f43f5e',
                    'knn': '#ec4899'
                }
                
                for m_k, m_v in models.items():
                    if hasattr(m_v, 'predict_proba'):
                        probs = m_v.predict_proba(x_test_scaled)
                        y_test_bin = (y_test == target_class_roc).astype(int)
                        y_score = probs[:, target_class_roc]
                        
                        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
                        roc_auc = auc(fpr, tpr)
                        
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr, mode='lines',
                            name=f'{m_dict[m_k]["name"]} (AUC={roc_auc:.3f})',
                            line=dict(color=line_colors.get(m_k, '#cccccc'), width=1.8)
                        ))
                fig_roc.add_shape(
                    type='line', line=dict(dash='dash', color='#475569', width=1.5),
                    x0=0, x1=1, y0=0, y1=1
                )
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#ffffff',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[-0.02, 1.02]),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[-0.02, 1.02]),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5,
                        bgcolor='rgba(15,23,42,0.6)', bordercolor='rgba(255,255,255,0.08)'
                    ),
                    margin=dict(l=30, r=20, t=40, b=30), height=350
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Test splits data not available for ROC curve calculation.")

elif page_selection == "ℹ About Project":
    # ------------------ ABOUT PROJECT PAGE ------------------
    st.markdown("<div class='main-title'>ℹ About the Project</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>Academic final-year project documentation and overview.</div>", unsafe_allow_html=True)
    
    col_ab1, col_ab2 = st.columns(2)
    
    with col_ab1:
        st.markdown("""
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
            <h3>📡 20 Sensor Features</h3>
            <div style="font-size: 0.88rem; line-height: 1.4; color: #4b5563; font-weight:550; max-height: 180px; overflow-y: auto;">
                1. <b>Timestamp</b> (Temporal reference index)<br>
                2. <b>Fault_Label</b> (Class indicator variable)<br>
                3. <b>Shaft RPM</b> (Propulsion core rotation speed)<br>
                4. <b>Engine Load</b> (Percentage engine output torque)<br>
                5. <b>Fuel Flow</b> (Fuel volumetric flow rate)<br>
                6. <b>Air Pressure</b> (Turbocharger boost air pressure)<br>
                7. <b>Ambient Temperature</b> (Machinery space temperature)<br>
                8. <b>Oil Temperature</b> (Lubricating system oil temperature)<br>
                9. <b>Oil Pressure</b> (Lubricating feed main pressure)<br>
                10-12. <b>Vibration X, Y, Z</b> (Radial/axial engine block displacement)<br>
                13-16. <b>Cylinder 1-4 Pressures</b> (Peak combustion cylinder pressures)<br>
                17-20. <b>Cylinder 1-4 Exhaust Temps</b> (Post-combustion gas temperatures)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_ab2:
        st.markdown("""
        <div class="white-card">
            <h3>🤖 Machine Learning Algorithms Used</h3>
            <p style="font-size: 0.9rem; line-height: 1.4; margin-top: 0.5rem; color: #4b5563; font-weight: 550;">
                - <b>Random Forest Classifier</b>: Ensemble model utilizing bootstrap bagging, chosen as the champion model due to high classification robustness.<br>
                - <b>XGBoost (Extreme Gradient Boosting)</b>: High-performance gradient boosted decision trees.<br>
                - <b>Decision Tree Classifier</b>: A hierarchical logic rules baseline.<br>
                - <b>Support Vector Machine (SVM)</b>: RBF-kernel high-dimensional distance separator.<br>
                - <b>K-Nearest Neighbors (KNN)</b>: Instance similarity neighborhood classification.<br>
                - <b>Logistic Regression</b>: Linear classification baseline model.
            </p>
        </div>
        
        <div class="white-card">
            <h3>📈 Prediction Process</h3>
            <ol style="font-size: 0.9rem; margin-top: 0.5rem; padding-left: 1.2rem; color: #4b5563; font-weight: 550;">
                <li>Telemetry values are parsed from sensors.</li>
                <li>Inputs are standardized using the pre-fit <code>scaler.pkl</code> weights.</li>
                <li>The standardized feature array is mapped by the active Classifier model.</li>
                <li>The target class with the highest posterior probability is output as the predicted fault.</li>
                <li>The corresponding risk severity and maintenance crew recommendation are loaded dynamically.</li>
            </ol>
        </div>
        
        <div class="white-card">
            <h3>🎯 Expected Output</h3>
            <p style="font-size: 0.9rem; line-height: 1.4; margin-top: 0.5rem; color: #4b5563; font-weight: 550;">
                Output comprises three diagnostic indexes: <b>Engine Health Score</b> (0-100%), <b>Predicted Fault</b> class label, and the <b>Maintenance Crew Checklist</b>.
            </p>
        </div>
        
        <div class="white-card">
            <h3>🚀 Future Scope</h3>
            <p style="font-size: 0.9rem; line-height: 1.4; margin-top: 0.5rem; color: #4b5563; font-weight: 550;">
                Integration of deep LSTM models for forecasting, edge deployment on PLC units, and real-time Kafka streaming processing.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center; border-top:1px solid rgba(255,255,255,0.08); margin-top:3rem; padding-top:1.5rem; padding-bottom:1.5rem;'>
    <div style='font-size:0.95rem; color:#38bdf8; font-weight:700;'>
        AI-Based Marine Engine Predictive Maintenance Using Sensor Data
    </div>
    <div style='font-size:0.8rem; color:#64748b; font-weight:600; margin-top:0.3rem;'>
        Developed using Python, Streamlit, Plotly, Scikit-learn | Final Year B.Tech Project
    </div>
</div>
""", unsafe_allow_html=True)
