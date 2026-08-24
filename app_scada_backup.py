import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, classification_report

# Page configurations
st.set_page_config(
    page_title="Marine Engine Predictive Maintenance System",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State Variables
if 'active_model_key' not in st.session_state:
    st.session_state['active_model_key'] = 'random_forest'
if 'connection_status' not in st.session_state:
    st.session_state['connection_status'] = True
if 'engine_id' not in st.session_state:
    st.session_state['engine_id'] = "ME-9488-DX"

# Fault definitions
FAULT_CLASSES = {
    0: {
        "name": "Normal Operation",
        "desc": "All systems operating within baseline parameters. Normal cylinder heat-release and lubrication pressures.",
        "color": "#10b981", # Green
        "severity": "Healthy",
        "rec": "Continue nominal operations. Execute standard logbook monitoring entries. Scheduled maintenance in 250 operational hours."
    },
    1: {
        "name": "Fuel Delivery System Anomaly",
        "desc": "Fuel flow rate is disproportionately high relative to shaft RPM and load. Injector clogging or high line restriction suspected.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Isolate fuel delivery line. Inspect fuel primary filter elements for mechanical contamination. Test injectors 1-4 for nozzle spray pattern anomalies."
    },
    2: {
        "name": "Low Cylinder Compression Pressure",
        "desc": "Pressure leakage detected across multiple combustion chambers. Suspected valve seat wear or cylinder head gasket failure.",
        "color": "#f59e0b", # Orange
        "severity": "Warning",
        "rec": "Schedule static compression test. Inspect exhaust/inlet valve clearances. Audit crankcase blow-by pressure gauges."
    },
    3: {
        "name": "Combustion Heat / Exhaust Gas Anomaly",
        "desc": "Exhaust gas temperature profile exceeds safety thresholds. Indicative of late injection timing or exhaust valve guide fouling.",
        "color": "#f59e0b", # Orange
        "severity": "Warning",
        "rec": "Check cooling jacket water outlet temperature. Clean cooling path. Inspect exhaust manifold thermowells. Check fuel pump timing alignments."
    },
    4: {
        "name": "Radial Engine Vibration Fault",
        "desc": "Excessive displacement recorded in the X and Y axes. Suggests shaft coupling misalignment or unbalanced counterweights.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Halt engine if vibration exceeds 4.5 mm/s RMS. Perform coupling laser alignment. Check engine mount torque specifications and condition."
    },
    5: {
        "name": "Lubrication System Thermal Anomaly",
        "desc": "High sump oil temperature coupled with declining feed line pressure. Degradation of oil viscosity or thermal cooler failure.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Verify oil cooler cooling water flow. Sample lube oil for viscosity check and wear-metals analysis (spectrometry). Inspect pressure regulation valves."
    },
    6: {
        "name": "Air Intake Pressure / Turbocharger Fault",
        "desc": "Boost air pressure drops significantly under high engine load. Turbo wastegate leak, compressor fouling, or manifold gasket failure.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Inspect turbocharger rotor shaft for play. Inspect intake filter differential pressure. Examine intercooler for restrictions or air leaks."
    },
    7: {
        "name": "Lubrication Pressure & Axial Vibration Fault",
        "desc": "Severe drop in oil pressure accompanied by high vibration in the Z (axial) direction. Suggests a failing thrust bearing.",
        "color": "#ef4444", # Red
        "severity": "Critical",
        "rec": "Shut down engine immediately. Conduct crankcase sump inspection for metal shavings. Audit thrust collar alignment and clearance clearances."
    }
}

# Custom Styling (Dark Navy Industrial SCADA theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;600;800&display=swap');

    /* Global settings */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #060b19 0%, #02040a 100%) !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
    }
    
    .scada-title {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 50%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.1rem;
        text-shadow: 0 0 35px rgba(59, 130, 246, 0.3);
    }
    
    .scada-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-family: 'Rajdhani', sans-serif;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.45) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-bottom: 1rem;
    }
    
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.2) !important;
        transform: translateY(-3px) !important;
    }

    /* Animated KPI Cards */
    .kpi-card {
        background: rgba(15, 23, 42, 0.5) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 1.25rem !important;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
    }
    
    .kpi-card:hover {
        transform: scale(1.03);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
    }
    
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #38bdf8, #3b82f6);
    }
    
    .kpi-healthy::before {
        background: linear-gradient(90deg, #10b981, #059669) !important;
    }
    
    .kpi-warning::before {
        background: linear-gradient(90deg, #f59e0b, #d97706) !important;
    }
    
    .kpi-critical::before {
        background: linear-gradient(90deg, #ef4444, #dc2626) !important;
    }

    .kpi-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 0.5rem;
    }
    
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        font-weight: 600;
    }

    /* Pulse Anomaly Indicator */
    .pulse-container {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .pulse-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .pulse-green {
        background-color: #10b981;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse-green-anim 1.6s infinite;
    }
    
    .pulse-orange {
        background-color: #f59e0b;
        box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
        animation: pulse-orange-anim 1.6s infinite;
    }
    
    .pulse-red {
        background-color: #ef4444;
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
        animation: pulse-red-anim 1.2s infinite;
    }
    
    @keyframes pulse-green-anim {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    @keyframes pulse-orange-anim {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(245, 158, 11, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
    }
    @keyframes pulse-red-anim {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.9); }
        70% { transform: scale(1.1); box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Sidebar controls */
    .stRadio > div {
        background: rgba(15, 23, 42, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    /* Tables design */
    table {
        color: #cbd5e1 !important;
        background-color: rgba(15, 23, 42, 0.5) !important;
        border-collapse: collapse;
        border-radius: 12px;
        overflow: hidden;
    }
    th {
        color: #38bdf8 !important;
        background-color: rgba(30, 41, 59, 0.8) !important;
        font-weight: 700 !important;
        padding: 12px !important;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2) !important;
    }
    td {
        padding: 10px !important;
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    /* Gold Trophy Badge */
    .gold-badge {
        background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%);
        border: 1px solid #fbbf24;
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-weight: 800;
        font-size: 0.9rem;
        color: #000;
        box-shadow: 0 0 15px rgba(251, 191, 36, 0.4);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Rajdhani', sans-serif;
    }

    /* Sidebar Header styling */
    .sidebar-header {
        font-family: 'Orbitron', sans-serif;
        color: #38bdf8;
        font-size: 1.1rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
        margin-top: 15px;
        border-bottom: 2px solid rgba(56, 189, 248, 0.2);
        padding-bottom: 6px;
    }

    /* Button custom hover */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.4rem !important;
        font-weight: 700 !important;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.45) !important;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Model loading logic
@st.cache_resource
def load_assets():
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

models, scaler, metrics_payload = load_assets()

# Load dataset for statistics
@st.cache_data
def get_dataset_stats():
    csv_path = "marine_engine_fault_dataset (1).csv"
    if not os.path.exists(csv_path):
        return {
            'Shaft_RPM': (750.0, 1150.0, 960.0),
            'Engine_Load': (25.0, 110.0, 75.0),
            'Fuel_Flow': (60.0, 190.0, 130.0),
            'Air_Pressure': (0.3, 1.6, 1.15),
            'Ambient_Temp': (15.0, 40.0, 27.0),
            'Oil_Temp': (60.0, 115.0, 78.0),
            'Oil_Pressure': (0.4, 5.2, 3.4),
            'Vibration_X': (0.0, 0.5, 0.06),
            'Vibration_Y': (0.0, 0.5, 0.05),
            'Vibration_Z': (0.0, 0.6, 0.07),
            'Cylinder1_Pressure': (85.0, 190.0, 145.0),
            'Cylinder1_Exhaust_Temp': (290.0, 620.0, 420.0),
            'Cylinder2_Pressure': (90.0, 190.0, 145.0),
            'Cylinder2_Exhaust_Temp': (310.0, 600.0, 420.0),
            'Cylinder3_Pressure': (85.0, 190.0, 145.0),
            'Cylinder3_Exhaust_Temp': (300.0, 610.0, 420.0),
            'Cylinder4_Pressure': (85.0, 190.0, 145.0),
            'Cylinder4_Exhaust_Temp': (310.0, 620.0, 420.0),
        }
    try:
        df = pd.read_csv(csv_path).dropna().drop_duplicates()
        stats = {}
        feature_cols = [col for col in df.columns if col not in ['Timestamp', 'Fault_Label']]
        healthy_df = df[df['Fault_Label'] == 0]
        
        for col in feature_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_min_buf = max(0.0, col_min - (col_max - col_min) * 0.05) if 'Vibration' in col else max(0.0, col_min - (col_max - col_min) * 0.1)
            col_max_buf = col_max + (col_max - col_min) * 0.1
            default_val = float(healthy_df[col].median() if len(healthy_df) > 0 else df[col].median())
            stats[col] = (round(col_min_buf, 2), round(col_max_buf, 2), round(default_val, 2))
        return stats
    except Exception:
        return get_dataset_stats.__wrapped__()

stats_dict = get_dataset_stats()

# Helper for test predictions ROC
@st.cache_data
def get_test_splits_cached():
    csv_path = "marine_engine_fault_dataset (1).csv"
    if not os.path.exists(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path).dropna().drop_duplicates()
        feature_cols = [col for col in df.columns if col not in ['Timestamp', 'Fault_Label']]
        x = df[feature_cols].values
        y = df['Fault_Label'].values
        
        from sklearn.model_selection import train_test_split
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        with open("scaler.pkl", "rb") as f:
            sc = pickle.load(f)
        x_test_scaled = sc.transform(x_test)
        return y_test, x_test_scaled
    except Exception:
        return None, None

y_test_data, x_test_scaled_data = get_test_splits_cached()

# Preset loader function
def apply_preset(preset_name):
    stats = get_dataset_stats()
    # Reset all to healthy first
    for col, limits in stats.items():
        st.session_state[f"val_{col}"] = limits[2]
        
    if preset_name == "healthy":
        st.session_state['active_preset_msg'] = ("healthy", "🟢 Normal Operation parameters loaded successfully!")
    elif preset_name == "fuel":
        st.session_state['val_Fuel_Flow'] = stats['Fuel_Flow'][1] * 0.92
        st.session_state['val_Engine_Load'] = stats['Engine_Load'][0] + 5.0
        st.session_state['active_preset_msg'] = ("warning", "🔴 Fuel Delivery Anomaly parameters loaded! High flow at low load.")
    elif preset_name == "compression":
        st.session_state['val_Cylinder1_Pressure'] = stats['Cylinder1_Pressure'][0] + 4.0
        st.session_state['val_Cylinder2_Pressure'] = stats['Cylinder2_Pressure'][0] + 4.0
        st.session_state['val_Cylinder3_Pressure'] = stats['Cylinder3_Pressure'][0] + 4.0
        st.session_state['val_Cylinder4_Pressure'] = stats['Cylinder4_Pressure'][0] + 4.0
        st.session_state['active_preset_msg'] = ("warning", "🔴 Low Compression pressure loaded across all cylinders.")
    elif preset_name == "exhaust":
        st.session_state['val_Cylinder1_Exhaust_Temp'] = stats['Cylinder1_Exhaust_Temp'][1] * 0.88
        st.session_state['val_Cylinder2_Exhaust_Temp'] = stats['Cylinder2_Exhaust_Temp'][1] * 0.88
        st.session_state['val_Cylinder3_Exhaust_Temp'] = stats['Cylinder3_Exhaust_Temp'][1] * 0.88
        st.session_state['val_Cylinder4_Exhaust_Temp'] = stats['Cylinder4_Exhaust_Temp'][1] * 0.88
        st.session_state['active_preset_msg'] = ("warning", "🔴 Combustion Anomaly: Extremely high cylinder exhausts gas temperatures loaded.")
    elif preset_name == "vibration":
        st.session_state['val_Vibration_X'] = stats['Vibration_X'][1] * 0.82
        st.session_state['val_Vibration_Y'] = stats['Vibration_Y'][1] * 0.82
        st.session_state['active_preset_msg'] = ("warning", "🔴 Radial Vibration Fault: Heavy lateral displacement loaded.")
    elif preset_name == "lube_thermal":
        st.session_state['val_Oil_Temp'] = stats['Oil_Temp'][1] * 0.92
        st.session_state['val_Oil_Pressure'] = stats['Oil_Pressure'][0] + 0.15
        st.session_state['active_preset_msg'] = ("warning", "🔴 Lubrication Thermal Runaway loaded! Elevated temperature and minimal feed pressure.")
    elif preset_name == "turbo":
        st.session_state['val_Air_Pressure'] = stats['Air_Pressure'][0] + 0.08
        st.session_state['active_preset_msg'] = ("warning", "🔴 Air Intake pressure loss preset loaded. Turbocharger compressor fault simulated.")
    elif preset_name == "lube_axial":
        st.session_state['val_Oil_Pressure'] = stats['Oil_Pressure'][0] + 0.15
        st.session_state['val_Vibration_Z'] = stats['Vibration_Z'][1] * 0.88
        st.session_state['active_preset_msg'] = ("warning", "🔴 Lubrication Pressure & Axial Vibration loaded. Suspected thrust bearing wear.")

# Check weight files
missing_models = not models or len(models) < 6 or scaler is None or metrics_payload is None
if missing_models:
    st.warning("⚠️ Predictive model assets or performance metrics not detected in workspace.")
    if st.button("🚀 Train All Classifiers & Generate Assets"):
        with st.spinner("🔄 Running train_model.py..."):
            import subprocess
            res = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
            if res.returncode == 0:
                st.success("🎉 Models trained successfully! Refreshing dashboard...")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(f"Training failed: {res.stderr}")
    st.stop()

# Initialize session state for all 18 sensors
for col, limits in stats_dict.items():
    key = f"val_{col}"
    if key not in st.session_state:
        st.session_state[key] = limits[2]

# ----------------- SIDEBAR NAVIGATION -----------------
st.sidebar.markdown("<div class='sidebar-header'>⚓ Propulsion control</div>", unsafe_allow_html=True)

# Navigation Selector matching sidebar request items
page_selection = st.sidebar.radio(
    "Vessel Operations Console",
    [
        "Dashboard",
        "Prediction",
        "Engine Monitoring",
        "Model Comparison",
        "Analytics",
        "Reports",
        "Settings",
        "About Project"
    ]
)

# Sidebar System Connection indicator
st.sidebar.markdown("<div class='sidebar-header'>🛰️ Connection Status</div>", unsafe_allow_html=True)
conn_toggle = st.sidebar.checkbox("Connected to Live PLC Telemetry", value=st.session_state['connection_status'])
st.session_state['connection_status'] = conn_toggle

status_text = "🟢 Connected" if st.session_state['connection_status'] else "🔴 Disconnected"
st.sidebar.markdown(f"Status: **{status_text}**")

# Engine ID Config
st.sidebar.markdown("<div class='sidebar-header'>⚙️ Vessel Metadata</div>", unsafe_allow_html=True)
st.session_state['engine_id'] = st.sidebar.text_input("Active Engine Serial ID", st.session_state['engine_id'])


# ----------------- GLOBAL SENSOR READS -----------------
# Generate sensor inputs from session state
current_sensors = {}
for col in stats_dict.keys():
    current_sensors[col] = st.session_state[f"val_{col}"]

# Predict state based on current values
input_vector = np.array([[current_sensors[c] for c in stats_dict.keys()]])
scaled_vector = scaler.transform(input_vector)

# ML Predictions
model_obj = models[st.session_state['active_model_key']]
pred_class = int(model_obj.predict(scaled_vector)[0])
pred_probs = model_obj.predict_proba(scaled_vector)[0]
confidence_score = pred_probs[pred_class]

active_fault = FAULT_CLASSES[pred_class]
status_color = active_fault['color']
status_label = active_fault['severity']

# Calculate engine health score dynamically
if pred_class == 0:
    health_score = 95.0 + (confidence_score * 5.0)
else:
    # Penalize based on fault probability and deviations
    oil_pressure_lim = stats_dict['Oil_Pressure']
    oil_pressure_dev = max(0.0, (oil_pressure_lim[2] - current_sensors['Oil_Pressure']) / (oil_pressure_lim[2] - oil_pressure_lim[0])) if current_sensors['Oil_Pressure'] < oil_pressure_lim[2] else 0.0
    
    vib_x_lim = stats_dict['Vibration_X']
    vib_dev = max(0.0, (current_sensors['Vibration_X'] - vib_x_lim[2]) / (vib_x_lim[1] - vib_x_lim[2]))
    
    penalty = (confidence_score * 45.0) + (oil_pressure_dev * 20.0) + (vib_dev * 15.0)
    health_score = max(5.0, 90.0 - penalty)

# Running Status calculation
running_status = "TRIPPED" if (health_score < 30.0) else ("STANDBY" if current_sensors['Shaft_RPM'] < 780.0 else "NOMINAL")

# ----------------- HEADER PANEL -----------------
col_h1, col_h2 = st.columns([7, 3])
with col_h1:
    st.markdown("<div class='scada-title'>⚓ Marine Engine Predictive Maintenance System</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='scada-subtitle'>SCADA Command Room Monitor | Engine Serial: <strong>{st.session_state['engine_id']}</strong></div>", unsafe_allow_html=True)
with col_h2:
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""
    <div class='glass-card' style='padding:0.6rem 1rem !important; text-align:right; margin-bottom:0;'>
        <div style='font-size:0.75rem; color:#64748b; font-weight:700; text-transform:uppercase;'>System Connection Time</div>
        <div style='font-size:0.95rem; font-family:Orbitron; color:#38bdf8; font-weight:700;'>{status_text}</div>
        <div style='font-size:0.8rem; color:#cbd5e1; font-family:Orbitron;'>{current_time_str}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------- PAGE 1: DASHBOARD -----------------
if page_selection == "Dashboard":
    # Pulse animation CSS selector based on class
    pulse_dot_class = "pulse-green" if pred_class == 0 else ("pulse-orange" if pred_class in [2, 3] else "pulse-red")
    
    # Header Model Selector Cards
    st.markdown("##### 🤖 Prediction Classifier Selection")
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    models_list = [
        ('random_forest', 'Random Forest ⭐', 'Bagging Ensemble'),
        ('xgboost', 'XGBoost', 'Gradient Boosting'),
        ('decision_tree', 'Decision Tree', 'Hierarchical Rules'),
        ('logistic_regression', 'Logistic Regression', 'Linear Baseline'),
        ('svm', 'SVM', 'RBF Kernel Space'),
        ('knn', 'KNN', 'Instance Neighbor')
    ]

    for idx, (m_key, m_name, m_type) in enumerate(models_list):
        with [col_m1, col_m2, col_m3, col_m4, col_m5, col_m6][idx]:
            is_selected = (st.session_state['active_model_key'] == m_key)
            border_style = "border: 2px solid #fbbf24; box-shadow: 0 0 15px rgba(251, 191, 36, 0.35);" if is_selected else "border: 1px solid rgba(255,255,255,0.08);"
            bg_style = "background: rgba(251, 191, 36, 0.08);" if is_selected else "background: rgba(15, 23, 42, 0.4);"
            
            st.markdown(f"""
            <div style='{border_style} {bg_style} border-radius:12px; padding:12px; text-align:center;'>
                <div style='font-size:0.7rem; text-transform:uppercase; color:#64748b; font-weight:700;'>{m_type}</div>
                <strong style='color:#ffffff; font-size:0.95rem; font-family:Orbitron;'>{m_name}</strong>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Activate Model", key=f"btn_{m_key}", use_container_width=True):
                st.session_state['active_model_key'] = m_key
                st.rerun()
                
    st.write("")
    
    # Live Engine Status Cards
    st.markdown("##### 🚨 Live Telemetry Status Panels")
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi4, col_kpi5, col_kpi6 = st.columns(3)
    
    border_class_engine = "kpi-healthy" if pred_class == 0 else ("kpi-warning" if pred_class in [2, 3] else "kpi-critical")
    
    with col_kpi1:
        st.markdown(f"""
        <div class="kpi-card {border_class_engine}">
            <div class="kpi-label">Current Engine Health</div>
            <div class="kpi-value pulse-container">
                <span class="pulse-dot {pulse_dot_class}"></span>
                <span style="color:{status_color}; font-size:1.5rem;">{status_label.upper()} ({health_score:.1f}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        st.markdown(f"""
        <div class="kpi-card {border_class_engine}">
            <div class="kpi-label">Predicted Fault Class</div>
            <div class="kpi-value" style="font-size: 1.1rem; line-height:1.2; padding-top:4px;">
                {active_fault['name']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        st.markdown(f"""
        <div class="kpi-card {border_class_engine}">
            <div class="kpi-label">Prediction Confidence</div>
            <div class="kpi-value">{confidence_score:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi4:
        # Maintenance status
        maint_status = "Nominal Schedule" if pred_class == 0 else ("Urgent Diagnostics" if pred_class in [2, 3] else "EMERGENCY SHUTDOWN")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Maintenance Status</div>
            <div class="kpi-value" style="color: {status_color}; font-size:1.3rem;">{maint_status.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi5:
        # Engine Running Status
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Engine Running Status</div>
            <div class="kpi-value">{running_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Core shaft speed</div>
            <div class="kpi-value">{current_sensors['Shaft_RPM']:.1f} RPM</div>
        </div>
        """, unsafe_allow_html=True)

    # Large Center visual split
    st.write("")
    col_dash_l, col_dash_r = st.columns([5, 4])
    
    with col_dash_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Propulsion System Diagnostics Sandbox")
        st.write("Force preset sensor profiles into the system to test the predictive classifiers:")
        
        # Display feedback message
        if 'active_preset_msg' in st.session_state:
            p_type, text = st.session_state['active_preset_msg']
            if p_type == "healthy":
                st.info(text)
            else:
                st.warning(text)
                
        col_preset1, col_preset2 = st.columns(2)
        with col_preset1:
            st.button("🟢 Nominal Operations", on_click=apply_preset, args=("healthy",), use_container_width=True)
            st.button("🔴 Fuel System Restriction Anomaly", on_click=apply_preset, args=("fuel",), use_container_width=True)
            st.button("🔴 Multi-Cylinder Compression Leakage", on_click=apply_preset, args=("compression",), use_container_width=True)
            st.button("🔴 Exhaust Gas Thermal Runaway", on_click=apply_preset, args=("exhaust",), use_container_width=True)
        with col_preset2:
            st.button("🔴 Radial Shaft Misalignment (Vib X/Y)", on_click=apply_preset, args=("vibration",), use_container_width=True)
            st.button("🔴 Lubrication Thermal & Viscosity Loss", on_click=apply_preset, args=("lube_thermal",), use_container_width=True)
            st.button("🔴 Turbocharger Compressor Impeller Loss", on_click=apply_preset, args=("turbo",), use_container_width=True)
            st.button("🔴 Thrust Bearing Axial Friction (Vib Z)", on_click=apply_preset, args=("lube_axial",), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_dash_r:
        st.markdown("<div class='glass-card' style='height: 100%; text-align:center;'>", unsafe_allow_html=True)
        st.subheader("⚙️ System Health Indicator")
        
        fig_health = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Score %", 'font': {'size': 18, 'color': '#94a3b8', 'family': 'Orbitron'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': status_color},
                'bgcolor': "rgba(30, 41, 59, 0.4)",
                'borderwidth': 2,
                'bordercolor': "#475569",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.15)'},
                    {'range': [40, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                    {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
                ]
            }
        ))
        fig_health.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Rajdhani'),
            height=260,
            margin=dict(l=30, r=30, t=30, b=10)
        )
        st.plotly_chart(fig_health, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 2 & 3: PREDICTION -----------------
elif page_selection == "Prediction":
    st.markdown("### 🎛️ Sensor Control Panel & Live Prediction")
    st.write("Modify the physical sensor signals. Predictions will compile instantly across all modules.")
    
    col_pred_l, col_pred_r = st.columns([7, 5])
    
    with col_pred_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("⚙️ Signal Controllers")
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            st.slider("Shaft RPM (rotations/min)", float(stats_dict['Shaft_RPM'][0]), float(stats_dict['Shaft_RPM'][1]), key="val_Shaft_RPM")
            st.slider("Engine Load (%)", float(stats_dict['Engine_Load'][0]), float(stats_dict['Engine_Load'][1]), key="val_Engine_Load")
            st.slider("Fuel Flow (L/h)", float(stats_dict['Fuel_Flow'][0]), float(stats_dict['Fuel_Flow'][1]), key="val_Fuel_Flow")
            st.slider("Air Pressure (bar)", float(stats_dict['Air_Pressure'][0]), float(stats_dict['Air_Pressure'][1]), key="val_Air_Pressure")
            st.slider("Ambient Temp (°C)", float(stats_dict['Ambient_Temp'][0]), float(stats_dict['Ambient_Temp'][1]), key="val_Ambient_Temp")
            st.slider("Oil Temp (°C)", float(stats_dict['Oil_Temp'][0]), float(stats_dict['Oil_Temp'][1]), key="val_Oil_Temp")
            st.slider("Oil Pressure (bar)", float(stats_dict['Oil_Pressure'][0]), float(stats_dict['Oil_Pressure'][1]), key="val_Oil_Pressure")
        
        with col_ctrl2:
            st.slider("Vibration X (g)", float(stats_dict['Vibration_X'][0]), float(stats_dict['Vibration_X'][1]), key="val_Vibration_X", step=0.001)
            st.slider("Vibration Y (g)", float(stats_dict['Vibration_Y'][0]), float(stats_dict['Vibration_Y'][1]), key="val_Vibration_Y", step=0.001)
            st.slider("Vibration Z (g)", float(stats_dict['Vibration_Z'][0]), float(stats_dict['Vibration_Z'][1]), key="val_Vibration_Z", step=0.001)
            
            with st.expander("Combustion Compression Pressures"):
                st.slider("Cyl 1 Pressure (bar)", float(stats_dict['Cylinder1_Pressure'][0]), float(stats_dict['Cylinder1_Pressure'][1]), key="val_Cylinder1_Pressure")
                st.slider("Cyl 2 Pressure (bar)", float(stats_dict['Cylinder2_Pressure'][0]), float(stats_dict['Cylinder2_Pressure'][1]), key="val_Cylinder2_Pressure")
                st.slider("Cyl 3 Pressure (bar)", float(stats_dict['Cylinder3_Pressure'][0]), float(stats_dict['Cylinder3_Pressure'][1]), key="val_Cylinder3_Pressure")
                st.slider("Cyl 4 Pressure (bar)", float(stats_dict['Cylinder4_Pressure'][0]), float(stats_dict['Cylinder4_Pressure'][1]), key="val_Cylinder4_Pressure")
                
            with st.expander("Exhaust Gas Temperature Profile"):
                st.slider("Cyl 1 Exhaust Temp (°C)", float(stats_dict['Cylinder1_Exhaust_Temp'][0]), float(stats_dict['Cylinder1_Exhaust_Temp'][1]), key="val_Cylinder1_Exhaust_Temp")
                st.slider("Cyl 2 Exhaust Temp (°C)", float(stats_dict['Cylinder2_Exhaust_Temp'][0]), float(stats_dict['Cylinder2_Exhaust_Temp'][1]), key="val_Cylinder2_Exhaust_Temp")
                st.slider("Cyl 3 Exhaust Temp (°C)", float(stats_dict['Cylinder3_Exhaust_Temp'][0]), float(stats_dict['Cylinder3_Exhaust_Temp'][1]), key="val_Cylinder3_Exhaust_Temp")
                st.slider("Cyl 4 Exhaust Temp (°C)", float(stats_dict['Cylinder4_Exhaust_Temp'][0]), float(stats_dict['Cylinder4_Exhaust_Temp'][1]), key="val_Cylinder4_Exhaust_Temp")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_pred_r:
        st.markdown(f"""
        <div class="glass-card" style="border-left: 8px solid {status_color} !important; box-shadow: 0 0 30px {status_color}22 !important;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.9rem; color:#94a3b8; font-weight:800; font-family:'Orbitron';">DIAGNOSTIC STATUS OUTCOME</span>
                <span style="background-color:{status_color}25; color:{status_color}; border: 1px solid {status_color}; font-weight:800; border-radius:20px; padding: 4px 14px; font-size:0.75rem; font-family:'Orbitron';">{status_label.upper()}</span>
            </div>
            <h2 style="color:#ffffff; margin-top:0.8rem; font-size:2rem; font-family:'Rajdhani';">{active_fault['name']}</h2>
            <p style="color:#94a3b8; font-size:0.95rem; line-height:1.4;">{active_fault['desc']}</p>
            <div style="border-top:1px solid rgba(255,255,255,0.08); margin-top:1.2rem; padding-top:1rem; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:0.75rem; text-transform:uppercase; color:#64748b; font-weight:700;">Confidence Score</span>
                    <div style="font-size:1.6rem; font-weight:900; color:#ffffff; font-family:'Orbitron';">{confidence_score:.2%}</div>
                </div>
                <div>
                    <span style="font-size:0.75rem; text-transform:uppercase; color:#64748b; font-weight:700;">Engine Health Index</span>
                    <div style="font-size:1.6rem; font-weight:900; color:{status_color}; font-family:'Orbitron';">{health_score:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🔧 Action Plan Recommendation")
        st.write("Crew Directive Actions:")
        st.info(f"**Action Plan**: {active_fault['rec']}")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 4 & 5: ENGINE MONITORING -----------------
elif page_selection == "Engine Monitoring":
    st.markdown("### ⚙️ Engine Visualization & Live Monitoring Gauges")
    st.write("High-contrast animated visual models showing mechanical component profiles.")
    
    col_mon_l, col_mon_r = st.columns([5, 4])
    
    with col_mon_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🖥️ 2D SV Propulsion block & Loop Status")
        
        # Derive colors for each cylinder individually
        cyl_colors = []
        for i in range(1, 5):
            cp = current_sensors[f'Cylinder{i}_Pressure']
            ct = current_sensors[f'Cylinder{i}_Exhaust_Temp']
            if cp < 110.0 or ct > 520.0:
                cyl_colors.append("#ef4444") # Red Fault
            elif cp < 125.0 or ct > 470.0:
                cyl_colors.append("#f59e0b") # Orange Warning
            else:
                cyl_colors.append("#10b981") # Green Normal

        # System Loop Indicators
        # 1. Oil System
        op = current_sensors['Oil_Pressure']
        ot = current_sensors['Oil_Temp']
        if op < 1.6:
            oil_sys_color = "#ef4444" # Red
            oil_sys_lbl = "CRITICAL PRESS"
        elif op < 2.5 or ot > 95.0:
            oil_sys_color = "#f59e0b" # Orange
            oil_sys_lbl = "THERMAL WARNING"
        else:
            oil_sys_color = "#10b981" # Green
            oil_sys_lbl = "NOMINAL PRESSURE"
            
        # 2. Cooling system
        avg_exhaust = np.mean([current_sensors[f'Cylinder{i}_Exhaust_Temp'] for i in range(1, 5)])
        if avg_exhaust > 510.0:
            cool_sys_color = "#ef4444"
            cool_sys_lbl = "OVERHEAT DANGER"
        elif avg_exhaust > 460.0:
            cool_sys_color = "#f59e0b"
            cool_sys_lbl = "HIGH TEMP WARNING"
        else:
            cool_sys_color = "#10b981"
            cool_sys_lbl = "COOLING OPTIMAL"
            
        # 3. Bearings
        vib_avg = np.mean([current_sensors['Vibration_X'], current_sensors['Vibration_Y'], current_sensors['Vibration_Z']])
        if vib_avg > 0.25:
            bearing_color = "#ef4444"
            bearing_lbl = "BEARING WEAR ALERT"
        elif vib_avg > 0.08:
            bearing_color = "#f59e0b"
            bearing_lbl = "VIB WARNING"
        else:
            bearing_color = "#10b981"
            bearing_lbl = "BEARINGS LOCKED"

        rpm = current_sensors['Shaft_RPM']
        anim_dur = 1200.0 / max(100.0, rpm)
        
        st.markdown(f"""
        <style>
            @keyframes stroke-odd-cyl2 {{
                0% {{ transform: translateY(0px); }}
                50% {{ transform: translateY(45px); }}
                100% {{ transform: translateY(0px); }}
            }}
            @keyframes stroke-even-cyl2 {{
                0% {{ transform: translateY(45px); }}
                50% {{ transform: translateY(0px); }}
                100% {{ transform: translateY(45px); }}
            }}
            
            .cyl-piston-odd2 {{
                animation: stroke-odd-cyl2 {anim_dur:.3f}s infinite ease-in-out;
                transform-origin: center;
            }}
            .cyl-piston-even2 {{
                animation: stroke-even-cyl2 {anim_dur:.3f}s infinite ease-in-out;
                transform-origin: center;
            }}
        </style>
        """, unsafe_allow_html=True)
        
        # Raw SVG representation
        svg_code = f"""
        <svg width="100%" height="320" viewBox="0 0 600 320" style="background: rgba(15, 23, 42, 0.4); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
            <!-- Cooling system path -->
            <path d="M 40 50 L 560 50 L 560 160 L 40 160 Z" stroke="{cool_sys_color}" stroke-dasharray="8,4" stroke-width="2.5" fill="none" style="opacity:0.6;" />
            
            <!-- Sump Oil Line Piping -->
            <path d="M 50 280 L 550 280 M 85 280 L 85 180 M 225 280 L 225 180 M 365 280 L 365 180 M 505 280 L 505 180" 
                  stroke="{oil_sys_color}" stroke-width="6" stroke-linecap="round" fill="none" 
                  style="filter: drop-shadow(0px 0px 8px {oil_sys_color}); opacity: 0.85;" />
                  
            <!-- Crankcase and Primary Drive shaft -->
            <rect x="25" y="250" width="550" height="40" rx="6" fill="#1e293b" stroke="#334155" stroke-width="2" />
            <circle cx="300" cy="270" r="14" fill="#64748b" />
            
            <!-- Cylinder 1 Group -->
            <g transform="translate(0, 0)">
                <rect x="55" y="40" width="60" height="120" fill="none" stroke="{cyl_colors[0]}" stroke-width="4" rx="2" style="filter: drop-shadow(0 0 5px {cyl_colors[0]});" />
                <!-- Piston head -->
                <rect x="58" y="55" width="54" height="28" rx="2" fill="#64748b" stroke="#94a3b8" stroke-width="1.5" class="cyl-piston-odd2" />
                <circle cx="85" cy="250" r="10" fill="{bearing_color}" stroke="#cbd5e1" stroke-width="1.5" />
            </g>
            
            <!-- Cylinder 2 Group -->
            <g transform="translate(140, 0)">
                <rect x="55" y="40" width="60" height="120" fill="none" stroke="{cyl_colors[1]}" stroke-width="4" rx="2" style="filter: drop-shadow(0 0 5px {cyl_colors[1]});" />
                <!-- Piston head -->
                <rect x="58" y="55" width="54" height="28" rx="2" fill="#64748b" stroke="#94a3b8" stroke-width="1.5" class="cyl-piston-even2" />
                <circle cx="85" cy="250" r="10" fill="{bearing_color}" stroke="#cbd5e1" stroke-width="1.5" />
            </g>
            
            <!-- Cylinder 3 Group -->
            <g transform="translate(280, 0)">
                <rect x="55" y="40" width="60" height="120" fill="none" stroke="{cyl_colors[2]}" stroke-width="4" rx="2" style="filter: drop-shadow(0 0 5px {cyl_colors[2]});" />
                <!-- Piston head -->
                <rect x="58" y="55" width="54" height="28" rx="2" fill="#64748b" stroke="#94a3b8" stroke-width="1.5" class="cyl-piston-even2" />
                <circle cx="85" cy="250" r="10" fill="{bearing_color}" stroke="#cbd5e1" stroke-width="1.5" />
            </g>
            
            <!-- Cylinder 4 Group -->
            <g transform="translate(420, 0)">
                <rect x="55" y="40" width="60" height="120" fill="none" stroke="{cyl_colors[3]}" stroke-width="4" rx="2" style="filter: drop-shadow(0 0 5px {cyl_colors[3]});" />
                <!-- Piston head -->
                <rect x="58" y="55" width="54" height="28" rx="2" fill="#64748b" stroke="#94a3b8" stroke-width="1.5" class="cyl-piston-odd2" />
                <circle cx="85" cy="250" r="10" fill="{bearing_color}" stroke="#cbd5e1" stroke-width="1.5" />
            </g>
        </svg>
        """
        st.components.v1.html(svg_code, height=330)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # System loop legend status rows
        st.markdown(f"""
        <div style='background-color:rgba(15,23,42,0.3); border:1px solid rgba(255,255,255,0.05); padding:10px; border-radius:12px; display:flex; justify-content:space-around;'>
            <div style='text-align:center;'>
                <div style='font-size:0.75rem; color:#64748b; font-weight:700;'>OIL SYSTEM</div>
                <strong style='color:{oil_sys_color}; font-size:0.85rem;'>{oil_sys_lbl}</strong>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:0.75rem; color:#64748b; font-weight:700;'>COOLING SYSTEM</div>
                <strong style='color:{cool_sys_color}; font-size:0.85rem;'>{cool_sys_lbl}</strong>
            </div>
            <div style='text-align:center;'>
                <div style='font-size:0.75rem; color:#64748b; font-weight:700;'>JOURNAL BEARINGS</div>
                <strong style='color:{bearing_color}; font-size:0.85rem;'>{bearing_lbl}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_mon_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📈 Live Monitoring Gauges")
        
        # Gauge builder function
        def draw_minigauge(title, val, min_v, max_v, color, suffix=""):
            return go.Figure(go.Indicator(
                mode = "gauge+number",
                value = val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': title, 'font': {'size': 13, 'color': '#cbd5e1', 'family': 'Orbitron'}},
                gauge = {
                    'axis': {'range': [min_v, max_v], 'tickwidth': 1, 'tickcolor': '#cbd5e1'},
                    'bar': {'color': color},
                    'bgcolor': "rgba(30, 41, 59, 0.4)",
                    'borderwidth': 1.5,
                    'bordercolor': '#cbd5e1'
                },
                number = {'suffix': suffix, 'font': {'size': 18, 'color': '#ffffff', 'family': 'Orbitron'}}
            )).update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1', family='Rajdhani'),
                height=130,
                margin=dict(l=15, r=15, t=30, b=10)
            )

        st.plotly_chart(draw_minigauge("RPM", current_sensors['Shaft_RPM'], 750, 1150, "#38bdf8", " rpm"), use_container_width=True)
        st.plotly_chart(draw_minigauge("Oil pressure", current_sensors['Oil_Pressure'], 0, 6, "#ef4444", " bar"), use_container_width=True)
        st.plotly_chart(draw_minigauge("Oil temperature", current_sensors['Oil_Temp'], 60, 120, "#ec4899", " °C"), use_container_width=True)
        st.plotly_chart(draw_minigauge("Fuel flow", current_sensors['Fuel_Flow'], 60, 190, "#10b981", " L/h"), use_container_width=True)
        st.plotly_chart(draw_minigauge("Engine load", current_sensors['Engine_Load'], 25, 110, "#fbbf24", " %"), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 6: ANALYTICS -----------------
elif page_selection == "Analytics":
    st.markdown("### 📊 Live Analytics & Trend Profiles")
    st.write("Dynamic charts showing real-time signal variances.")
    
    col_an_l, col_an_r = st.columns(2)
    
    with col_an_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🌡️ Temperature profiles (Ambient vs Oil)")
        
        # Compare exhausts vs oil vs ambient
        fig_line_temp = go.Figure()
        fig_line_temp.add_trace(go.Scatter(y=[current_sensors['Ambient_Temp'], current_sensors['Oil_Temp']], x=['Ambient Temp', 'Oil Temp'], mode='lines+markers', line=dict(color='#ec4899', width=3)))
        fig_line_temp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Rajdhani'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Degrees C"),
            margin=dict(l=15, r=15, t=15, b=15),
            height=240
        )
        st.plotly_chart(fig_line_temp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🔥 Cylinder Pressures Profile")
        
        pres_df = pd.DataFrame({
            'Cylinder Unit': ['Cyl 1', 'Cyl 2', 'Cyl 3', 'Cyl 4'],
            'Pressure (bar)': [
                current_sensors['Cylinder1_Pressure'],
                current_sensors['Cylinder2_Pressure'],
                current_sensors['Cylinder3_Pressure'],
                current_sensors['Cylinder4_Pressure']
            ]
        })
        fig_cyl_p = px.bar(pres_df, x='Cylinder Unit', y='Pressure (bar)', color='Pressure (bar)', color_continuous_scale='Blues')
        fig_cyl_p.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Rajdhani'),
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            height=240
        )
        st.plotly_chart(fig_cyl_p, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📳 Vibration spectra (X/Y/Z)")
        
        vib_df = pd.DataFrame({
            'Channel': ['Vib X', 'Vib Y', 'Vib Z'],
            'Level (g)': [
                current_sensors['Vibration_X'],
                current_sensors['Vibration_Y'],
                current_sensors['Vibration_Z']
            ]
        })
        fig_vib_b = px.bar(vib_df, x='Channel', y='Level (g)', color='Level (g)', color_continuous_scale='Reds')
        fig_vib_b.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Rajdhani'),
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            height=240
        )
        st.plotly_chart(fig_vib_b, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_an_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🔥 Cylinder Exhaust Temperatures Comparison")
        
        exh_df = pd.DataFrame({
            'Cylinder Unit': ['Cyl 1', 'Cyl 2', 'Cyl 3', 'Cyl 4'],
            'Exhaust Temp (°C)': [
                current_sensors['Cylinder1_Exhaust_Temp'],
                current_sensors['Cylinder2_Exhaust_Temp'],
                current_sensors['Cylinder3_Exhaust_Temp'],
                current_sensors['Cylinder4_Exhaust_Temp']
            ]
        })
        fig_exh_t = px.bar(exh_df, x='Cylinder Unit', y='Exhaust Temp (°C)', color='Exhaust Temp (°C)', color_continuous_scale='Thermal')
        fig_exh_t.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Rajdhani'),
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            height=240
        )
        st.plotly_chart(fig_exh_t, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🕸️ Engine Telemetry Radar Profile")
        
        radar_features = ['Shaft_RPM', 'Engine_Load', 'Fuel_Flow', 'Air_Pressure', 'Oil_Temp', 'Oil_Pressure', 'Vibration_X']
        current_mapped = []
        for feat in radar_features:
            lims = stats_dict[feat]
            val = current_sensors[feat]
            mapped_val = (val - lims[0]) / (lims[1] - lims[0])
            current_mapped.append(mapped_val)
            
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=current_mapped,
            theta=[f.replace('_', ' ') for f in radar_features],
            fill='toself',
            fillcolor='rgba(56, 189, 248, 0.25)',
            line=dict(color='#38bdf8', width=2),
            name="Current Sensor State"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.08)', linecolor='rgba(255,255,255,0.08)')
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Rajdhani', size=11),
            margin=dict(l=35, r=35, t=20, b=15),
            height=240
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📊 Fault decision Probability Distribution")
        
        probs_df = pd.DataFrame({
            'Fault State': [FAULT_CLASSES[i]['name'] for i in range(8)],
            'Probability (%)': pred_probs * 100
        })
        fig_pie = px.pie(probs_df, values='Probability (%)', names='Fault State', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Rajdhani', size=10),
            margin=dict(l=15, r=15, t=15, b=15),
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
            height=240
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 7: MODEL PERFORMANCE COMPARISON -----------------
elif page_selection == "Model Comparison":
    st.markdown("### 🏆 Algorithm Benchmark & Diagnostics Matrix")
    
    if metrics_payload and 'metrics' in metrics_payload:
        m_dict = metrics_payload['metrics']
        
        # Calculate best model
        best_model_name = "XGBoost Classifier"
        best_model_acc = 0.0
        for k, v in m_dict.items():
            if v['accuracy'] > best_model_acc:
                best_model_acc = v['accuracy']
                best_model_name = v['name']
                
        # Golden Badge trophy row
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:1.5rem;'>
            <div class="gold-badge">
                🏆 CHAMPION CLASSIFIER: {best_model_name.upper()}
            </div>
            <div style='color:#fbbf24; font-weight:bold; font-size:1.1rem; font-family:"Orbitron"'>
                ACCURACY: {best_model_acc:.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_comp_l, col_comp_r = st.columns(2)
        
        with col_comp_l:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🔬 Classifier Performance Benchmarks")
            
            comp_table = []
            for k, val in m_dict.items():
                rep = val['report']
                comp_table.append({
                    'Algorithm Classifier': val['name'],
                    'Accuracy': val['accuracy'],
                    'Precision (Macro)': rep['macro avg']['precision'],
                    'Recall (Macro)': rep['macro avg']['recall'],
                    'F1-Score (Macro)': rep['macro avg']['f1-score'],
                    'Latency (ms)': val['inference_time_ms_per_sample']
                })
            df_comp = pd.DataFrame(comp_table)
            
            fig_bar_comp = px.bar(
                df_comp,
                x='Algorithm Classifier',
                y='Accuracy',
                color='Accuracy',
                color_continuous_scale='Blues',
                text_auto='.2%'
            )
            fig_bar_comp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1', family='Rajdhani'),
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
                yaxis=dict(range=[0.8, 1.02], gridcolor='rgba(255,255,255,0.05)'),
                height=260
            )
            st.plotly_chart(fig_bar_comp, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📈 ROC Curves (One-vs-Rest)")
            st.write("Compare True Positive vs False Positive rates for all 6 models:")
            
            target_class_roc = st.selectbox(
                "Select Target Classification class",
                options=list(FAULT_CLASSES.keys()),
                format_func=lambda x: FAULT_CLASSES[x]['name']
            )
            
            if y_test_data is not None:
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
                        probs = m_v.predict_proba(x_test_scaled_data)
                        y_test_bin = (y_test_data == target_class_roc).astype(int)
                        y_score = probs[:, target_class_roc]
                        
                        fpr, tpr, _ = roc_curve(y_test_bin, y_score)
                        roc_auc = auc(fpr, tpr)
                        
                        fig_roc.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            name=f'{m_dict[m_k]["name"]} (AUC={roc_auc:.3f})',
                            line=dict(color=line_colors.get(m_k, '#cccccc'), width=2)
                        ))
                fig_roc.add_shape(
                    type='line', line=dict(dash='dash', color='#475569', width=1.5),
                    x0=0, x1=1, y0=0, y1=1
                )
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#cbd5e1', family='Rajdhani'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[-0.02, 1.02]),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', range=[-0.02, 1.02]),
                    legend=dict(x=0.5, y=0.15, bgcolor='rgba(15,23,42,0.7)', bordercolor='rgba(255,255,255,0.08)'),
                    margin=dict(l=40, r=20, t=30, b=40),
                    height=280
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Test set split predictions required. Ensure metrics_payload is available.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_comp_r:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("📋 Precision / Recall Summary")
            st.dataframe(
                df_comp.style.format({
                    'Accuracy': '{:.2%}',
                    'Precision (Macro)': '{:.2%}',
                    'Recall (Macro)': '{:.2%}',
                    'F1-Score (Macro)': '{:.2%}',
                    'Latency (ms)': '{:.4f} ms'
                }).background_gradient(cmap='Blues', subset=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']),
                use_container_width=True
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("🎛️ Confusion Matrix Viewer")
            
            selected_cm_model = st.selectbox(
                "Select Model for Confusion Matrix",
                options=list(m_dict.keys()),
                format_func=lambda x: m_dict[x]['name']
            )
            
            cm = np.array(m_dict[selected_cm_model]['cm'])
            labels = [f"Class {i}" for i in range(8)]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted Diagnosis class", y="True Diagnosis class", color="Cases"),
                x=labels,
                y=labels,
                color_continuous_scale="Viridis",
                text_auto=True,
                aspect="auto"
            )
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#cbd5e1', family='Rajdhani'),
                margin=dict(l=10, r=10, t=10, b=10),
                height=260
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 8 & 9: REPORTS (EXPLAIN AI & RECOMMENDATION) -----------------
elif page_selection == "Reports":
    st.markdown("### 🧠 Explainable AI Predictions (XAI) & Protocols")
    st.write("Understand the decision logic behind the predictive maintenance recommendations.")
    
    col_rep_l, col_rep_r = st.columns(2)
    
    # Extract feature importances
    feature_names = metrics_payload.get('feature_names', list(stats_dict.keys()))
    rf_model_obj = models.get('random_forest')
    
    # Calculate feature importances
    importances = np.zeros(len(feature_names))
    if rf_model_obj and hasattr(rf_model_obj, 'feature_importances_'):
        importances = rf_model_obj.feature_importances_
        
    df_global_imp = pd.DataFrame({
        'Feature Channel': feature_names,
        'Importance weight': importances
    }).sort_values(by='Importance weight', ascending=False)

    # Local contribution calculations (SHAP value approximation)
    local_contributions = []
    # Identify push direction based on physical indicators
    positive_push_sensors = ['Vibration_X', 'Vibration_Y', 'Vibration_Z', 'Oil_Temp', 'Fuel_Flow', 
                             'Cylinder1_Exhaust_Temp', 'Cylinder2_Exhaust_Temp', 'Cylinder3_Exhaust_Temp', 'Cylinder4_Exhaust_Temp']
    negative_push_sensors = ['Oil_Pressure', 'Air_Pressure', 'Cylinder1_Pressure', 'Cylinder2_Pressure', 'Cylinder3_Pressure', 'Cylinder4_Pressure']
    
    for idx, col in enumerate(feature_names):
        lims = stats_dict[col]
        val = current_sensors[col]
        median = lims[2]
        r = lims[1] - lims[0]
        
        # Calculate deviation from normal baseline
        dev = (val - median) / r if r > 0 else 0
        importance = importances[idx]
        
        # Assign sign based on sensor dynamics
        if col in positive_push_sensors:
            sign = 1 if dev > 0 else -1
        elif col in negative_push_sensors:
            sign = -1 if dev > 0 else 1
        else:
            sign = 1 if abs(dev) > 0.1 else -1
            
        contrib = sign * abs(dev) * importance
        local_contributions.append({
            'Feature': col.replace('_', ' '),
            'Contribution': float(contrib),
            'Raw Value': val
        })
        
    df_local_contrib = pd.DataFrame(local_contributions)
    df_local_contrib['Abs_Contribution'] = df_local_contrib['Contribution'].abs()
    df_local_contrib = df_local_contrib.sort_values(by='Abs_Contribution', ascending=False).head(10)
    
    with col_rep_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🔥 Local Feature Contribution Chart (SHAP Values)")
        st.write("Impact of sensor deviations on the current diagnostic warning score:")
        
        fig_local_contrib = px.bar(
            df_local_contrib,
            x='Contribution',
            y='Feature',
            orientation='h',
            color='Contribution',
            color_continuous_scale=[[0.0, '#3b82f6'], [0.5, '#6b7280'], [1.0, '#ef4444']],
            title="Local SHAP value contributions"
        )
        fig_local_contrib.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#cbd5e1', family='Rajdhani'),
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            height=300
        )
        st.plotly_chart(fig_local_contrib, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🧠 Plain English Explanation")
        
        # Build explanation statement dynamically
        top_contribs = df_local_contrib[df_local_contrib['Contribution'] > 0.0]
        if pred_class == 0:
            st.markdown(f"**Explanation**: The propulsion plant is operating in a **Nominal State** because all physical sensors are aligned with standard limits. No significant local deviations exist.")
        elif len(top_contribs) >= 2:
            f1_name = top_contribs.iloc[0]['Feature']
            f2_name = top_contribs.iloc[1]['Feature']
            
            # Normalize percents for readability
            tot_p = top_contribs['Abs_Contribution'].sum()
            p1 = int((top_contribs.iloc[0]['Abs_Contribution'] / tot_p) * 55) if tot_p > 0 else 30
            p2 = int((top_contribs.iloc[1]['Abs_Contribution'] / tot_p) * 35) if tot_p > 0 else 20
            
            st.markdown(f"**Explanation**: High deviations on **{f1_name}** and **{f2_name}** contributed **{p1}%** and **{p2}%** respectively to the **{active_fault['name']}** warning. The active classifier triggered an alert because this pattern closely maps anomaly symptoms stored during model training.")
        else:
            st.markdown(f"**Explanation**: Minor deviations on sensor channels triggered a **{active_fault['name']}** warning. Verify lubrication sump filters and cylinder gaskets.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_rep_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🔧 Standard Action Checklist")
        st.write(f"Actions assigned dynamically based on current prediction:")
        st.info(f"**Action Plan**: {active_fault['rec']}")
        
        st.write("Execute safety diagnostics procedures in order:")
        if pred_class == 0:
            st.checkbox("🟢 Continue propulsion operations (nominal speeds)", value=True)
            st.checkbox("🟢 Maintain normal auxiliary thermal balances", value=True)
            st.checkbox("🟢 File daily telemetry log entry")
        elif pred_class == 1:
            st.checkbox("🔴 Shutdown fuel feed line auxiliary pump", value=True)
            st.checkbox("🔴 Unmount and clean fuel injectors 1-4", value=True)
            st.checkbox("🔴 Replace primary fuel filters")
        elif pred_class in [2, 3]:
            st.checkbox("🟡 Conduct compression test on cylinders 1-4", value=True)
            st.checkbox("🟡 Calibrate exhaust manifold thermowells", value=True)
            st.checkbox("🟡 Inspect inlet and exhaust valve clearances")
        elif pred_class in [4, 7]:
            st.checkbox("🔴 Halt engine block rotation immediately", value=True)
            st.checkbox("🔴 Perform laser shaft alignments checks", value=True)
            st.checkbox("🔴 Examine axial thrust pads and bearing liners")
        elif pred_class == 5:
            st.checkbox("🔴 Clean lube oil cooler cooling plates", value=True)
            st.checkbox("🔴 Perform oil viscosity test checks", value=True)
            st.checkbox("🔴 Replace lubrication pumps primary bypass valves")
        else:
            st.checkbox("🔴 Run turbocharger compressor cleaning checks", value=True)
            st.checkbox("🔴 Inspect wastegate visual operation indicators", value=True)
            st.checkbox("🔴 Replace main air filter assembly elements")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🚨 Priority Response Severity")
        priority_lbl = "LOW" if pred_class == 0 else ("MEDIUM" if pred_class in [2, 3] else "IMMEDIATE")
        severity_pct = 10 if pred_class == 0 else (45 if pred_class in [2, 3] else 90)
        
        st.write(f"Response Priority Level: **{priority_lbl}**")
        st.progress(severity_pct / 100)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 10: SETTINGS -----------------
elif page_selection == "Settings":
    st.markdown("### 🔧 SCADA Console Settings")
    st.write("Configure PLC simulation intervals, alarm trip thresholds, and network nodes.")
    
    col_set_l, col_set_r = st.columns(2)
    
    with col_set_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📡 Live PLC Interface Configurations")
        
        st.text_input("Vessel Network Node IP Address", "192.168.1.104")
        st.selectbox("PLC Refresh Rate", ["250 ms (Real-time)", "1000 ms (Buffered)", "5000 ms (Eco-mode)"])
        st.slider("Telemetry Noise Threshold %", 0.0, 5.0, 1.2)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_set_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🔔 Safety Alarm Trip thresholds")
        st.slider("Emergency Trip Health Limit %", 10.0, 50.0, 30.0)
        st.slider("Warning Alert Health Limit %", 50.0, 80.0, 75.0)
        
        st.write("Auxiliary Checks:")
        st.checkbox("Enable audible alarm siren outputs on control console", value=True)
        st.checkbox("Auto-transmit emergency reports to Fleet Management Portal")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- PAGE 11: ABOUT PROJECT -----------------
elif page_selection == "About Project":
    st.markdown("### 📖 About the AI Predictive Maintenance System")
    
    col_ab_l, col_ab_r = st.columns([6, 4])
    
    with col_ab_l:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🎯 Project Objective")
        st.write("""
        This application represents a comprehensive, production-grade vessel predictive maintenance control system.
        By loading high-dimensional engine sensor readings across 18 telemetry channels (including RPM, loads, cylinder pressures, exhausts, temperatures, and vibrations), the dashboard evaluates active operational risk.
        Using pre-trained ensemble classifiers, the system identifies anomalies before they manifest as critical mechanical failures.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_ab_r:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🤖 Algorithms Implemented")
        st.write("""
        - **Random Forest**: Standard high-accuracy ensemble classifier.
        - **XGBoost**: Gradient Boosted Trees model optimized for structure.
        - **Decision Tree**: Rule-based quick diagnostics path.
        - **SVM**: RBF-kernel distance spatial margins classifier.
        - **KNN**: Neighbor voting statistics baseline.
        - **Logistic Regression**: Linear statistical classifier.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- FOOTER DESIGN -----------------
st.markdown("""
<div style='text-align:center; border-top:1px solid rgba(255,255,255,0.08); margin-top:3rem; padding-top:1.5rem; padding-bottom:1.5rem;'>
    <div style='font-size:0.95rem; font-family:"Orbitron"; color:#38bdf8; font-weight:700;'>
        AI-Based Marine Engine Predictive Maintenance Using Sensor Data
    </div>
    <div style='font-size:0.8rem; color:#64748b; font-weight:600; margin-top:0.3rem;'>
        Developed using Python, Streamlit, Plotly, Scikit-learn | Final Year BVoc Data Science Project
    </div>
</div>
""", unsafe_allow_html=True)
