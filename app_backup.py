import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Marine Engine Health & Multi-Model Diagnostics",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fault Labels Definitions
FAULT_CLASSES = {
    0: {
        "name": "Normal Operation",
        "desc": "All engine systems are operating within standard parameters. No anomalies detected.",
        "color": "#10B981", # Green
        "severity": "Healthy"
    },
    1: {
        "name": "Fuel Delivery System Anomaly",
        "desc": "Detected abnormal fuel flow rate relative to shaft speed and load. This could indicate a fuel line restriction, injector clog, or fuel pump failure.",
        "color": "#EF4444", # Red
        "severity": "Critical"
    },
    2: {
        "name": "Low Cylinder Compression Pressure",
        "desc": "Detected low pressure levels across multiple cylinders. This indicates possible piston ring wear, valve leakage, or cylinder head gasket failure.",
        "color": "#F59E0B", # Orange
        "severity": "Warning"
    },
    3: {
        "name": "Combustion Heat / Exhaust Gas Anomaly",
        "desc": "Abnormally high exhaust gas temperatures across cylinders. This points to cooling jacket fouling, exhaust manifold blockage, or late fuel injection timing.",
        "color": "#F59E0B", # Orange
        "severity": "Warning"
    },
    4: {
        "name": "Radial Engine Vibration Fault",
        "desc": "Excessive vibration levels detected in the X and Y axes. Likely caused by shaft misalignment, unbalanced rotating masses, or damaged radial bearings.",
        "color": "#EF4444", # Red
        "severity": "Critical"
    },
    5: {
        "name": "Lubrication System Thermal Anomaly",
        "desc": "High oil temperature paired with declining oil pressure. Indicates lubricant degradation, oil cooler failure, or bearing wear.",
        "color": "#EF4444", # Red
        "severity": "Critical"
    },
    6: {
        "name": "Air Intake Pressure / Turbocharger Fault",
        "desc": "Abnormally low air intake pressure. Suggests a turbocharger wastegate leak, compressor fouling, or intake manifold leakage.",
        "color": "#EF4444", # Red
        "severity": "Critical"
    },
    7: {
        "name": "Lubrication Pressure & Axial Vibration Fault",
        "desc": "Low oil pressure accompanied by high vibration in the Z (axial) direction. Suggests a failing thrust bearing or crankshaft thrust collar issues.",
        "color": "#EF4444", # Red
        "severity": "Critical"
    }
}

# Custom Premium CSS Styling
st.markdown("""
<style>
    /* Premium Grid CSS styling */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0d131f 0%, #030712 100%);
        color: #f3f4f6;
    }
    
    /* Table contrast styling */
    table {
        color: #e6edf3 !important;
        background-color: #111827 !important;
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
    }
    th {
        color: #60a5fa !important;
        background-color: #1f2937 !important;
        font-weight: 700 !important;
        padding: 10px !important;
    }
    td {
        color: #d1d5db !important;
        padding: 8px !important;
    }
    tr:nth-child(even) {
        background-color: #1f2937 !important;
    }
    
    /* Title styling with glowing text */
    .main-title {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.15);
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Glowing metric card */
    .metric-card {
        background: rgba(17, 24, 39, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.3rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #3b82f6;
        box-shadow: 0 12px 40px 0 rgba(59, 130, 246, 0.2);
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 0.35rem;
    }
    
    .metric-desc {
        font-size: 0.75rem;
        color: #60a5fa;
        margin-top: 0.35rem;
    }

    /* Diagnosis result card with glassmorphism */
    .diag-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    }
    
    .feature-group-header {
        color: #60a5fa;
        font-size: 1.15rem;
        font-weight: 700;
        border-bottom: 2px solid rgba(59, 130, 246, 0.2);
        padding-bottom: 0.4rem;
        margin-top: 1.4rem;
        margin-bottom: 0.7rem;
        letter-spacing: 0.02em;
    }
    
    /* Sidebar styling */
    .nav-header {
        font-size: 0.85rem;
        color: #9ca3af;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
        margin-top: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
    }
    
    /* Custom Streamlit Button Styling */
    div.stButton > button {
        background: rgba(31, 41, 55, 0.7) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
        text-align: center;
    }
    div.stButton > button:hover {
        background: #1e40af !important;
        color: #ffffff !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 0 18px rgba(59, 130, 246, 0.4) !important;
        transform: translateY(-2px);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
</style>
""", unsafe_allow_html=True)

# Load all models, scaler, and metrics
@st.cache_resource
def load_all_assets():
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
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            
    models = {}
    for key, filename in model_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    models[key] = pickle.load(f)
            except Exception as e:
                st.error(f"Error loading model {key}: {e}")
                
    metrics = None
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "rb") as f:
                metrics = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
            
    return models, scaler, metrics

models, scaler, metrics_payload = load_all_assets()

# Preset loader function
def load_preset(preset_name):
    stats = get_dataset_stats()
    # Reset all to healthy first
    for col, limits in stats.items():
        st.session_state[f"slider_{col}"] = limits[2]
        
    if preset_name == "healthy":
        st.session_state['preset_message'] = ("healthy", "🟢 Normal Operation parameters loaded successfully!")
    elif preset_name == "fuel":
        st.session_state['slider_Fuel_Flow'] = stats['Fuel_Flow'][1] * 0.90
        st.session_state['slider_Engine_Load'] = stats['Engine_Load'][0] + 5.0
        st.session_state['preset_message'] = ("warning", "🔴 Fuel Delivery Anomaly parameters loaded! Fuel flow is elevated relative to load.")
    elif preset_name == "compression":
        st.session_state['slider_Cylinder1_Pressure'] = stats['Cylinder1_Pressure'][0] + 5.0
        st.session_state['slider_Cylinder2_Pressure'] = stats['Cylinder2_Pressure'][0] + 5.0
        st.session_state['slider_Cylinder3_Pressure'] = stats['Cylinder3_Pressure'][0] + 5.0
        st.session_state['slider_Cylinder4_Pressure'] = stats['Cylinder4_Pressure'][0] + 5.0
        st.session_state['preset_message'] = ("warning", "🔴 Low Compression pressure parameters loaded across cylinders.")
    elif preset_name == "exhaust":
        st.session_state['slider_Cylinder1_Exhaust_Temp'] = stats['Cylinder1_Exhaust_Temp'][1] * 0.90
        st.session_state['slider_Cylinder2_Exhaust_Temp'] = stats['Cylinder2_Exhaust_Temp'][1] * 0.90
        st.session_state['slider_Cylinder3_Exhaust_Temp'] = stats['Cylinder3_Exhaust_Temp'][1] * 0.90
        st.session_state['slider_Cylinder4_Exhaust_Temp'] = stats['Cylinder4_Exhaust_Temp'][1] * 0.90
        st.session_state['preset_message'] = ("warning", "🔴 High Exhaust Gas Temperature parameters loaded across cylinders.")
    elif preset_name == "vibration":
        st.session_state['slider_Vibration_X'] = stats['Vibration_X'][1] * 0.85
        st.session_state['slider_Vibration_Y'] = stats['Vibration_Y'][1] * 0.85
        st.session_state['preset_message'] = ("warning", "🔴 Radial Vibration anomaly loaded! Vibration in X/Y axes is elevated.")
    elif preset_name == "lube_thermal":
        st.session_state['slider_Oil_Temp'] = stats['Oil_Temp'][1] * 0.90
        st.session_state['slider_Oil_Pressure'] = stats['Oil_Pressure'][0] + 0.2
        st.session_state['preset_message'] = ("warning", "🔴 Lubrication Thermal Runaway parameters loaded! High oil temperature paired with low pressure.")
    elif preset_name == "turbo":
        st.session_state['slider_Air_Pressure'] = stats['Air_Pressure'][0] + 0.1
        st.session_state['preset_message'] = ("warning", "🔴 Air Intake Pressure anomaly loaded! Turbocharger boost pressure is abnormally low.")
    elif preset_name == "lube_axial":
        st.session_state['slider_Oil_Pressure'] = stats['Oil_Pressure'][0] + 0.2
        st.session_state['slider_Vibration_Z'] = stats['Vibration_Z'][1] * 0.85
        st.session_state['preset_message'] = ("warning", "🔴 Lubrication Pressure & Axial Vibration anomaly loaded! Thrust collar/bearing friction elevated.")


# Load dataset for statistics
@st.cache_data
def get_dataset_stats():
    csv_path = "marine_engine_fault_dataset (1).csv"
    if not os.path.exists(csv_path):
        # Fallback values
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
        df = pd.read_csv(csv_path)
        df = df.dropna().drop_duplicates()
        
        stats = {}
        feature_cols = [col for col in df.columns if col not in ['Timestamp', 'Fault_Label']]
        
        # Median of healthy class (Fault_Label == 0)
        healthy_df = df[df['Fault_Label'] == 0]
        
        for col in feature_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            # Add padding buffers to sliders
            col_min_buf = max(0.0, col_min - (col_max - col_min) * 0.05) if 'Vibration' in col else max(0.0, col_min - (col_max - col_min) * 0.1)
            col_max_buf = col_max + (col_max - col_min) * 0.1
            
            default_val = float(healthy_df[col].median() if len(healthy_df) > 0 else df[col].median())
            stats[col] = (round(col_min_buf, 2), round(col_max_buf, 2), round(default_val, 2))
            
        return stats
    except Exception as e:
        st.warning(f"Error loading stats from CSV, using fallbacks: {e}")
        return get_dataset_stats.__wrapped__()

# Training function triggerable from UI
def trigger_training():
    with st.spinner("🔄 Preprocessing dataset & training all 6 classifiers..."):
        try:
            import subprocess
            res = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
            if res.returncode == 0:
                st.success("🎉 All 6 models retrained and metrics generated successfully!")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error(f"Training failed:\n{res.stderr}")
        except Exception as e:
            st.error(f"Failed to launch training script: {e}")

# Sidebar Header
st.sidebar.markdown("<div class='nav-header'>🚢 Diagnostics Hub</div>", unsafe_allow_html=True)
app_mode = st.sidebar.radio(
    "Select Workstation",
    ["📖 Project Info & Documentation", "📊 Model Comparison & Analytics", "🔍 Single Engine Diagnostics", "📁 Batch File Diagnostics"]
)

# Active Model Selector
st.sidebar.markdown("<div class='nav-header'>🤖 Predictive Model</div>", unsafe_allow_html=True)
model_options = {
    'logistic_regression': 'Logistic Regression (Baseline)',
    'random_forest': 'Random Forest Classifier',
    'xgboost': 'XGBoost Classifier',
    'decision_tree': 'Decision Tree Classifier',
    'svm': 'Support Vector Machine (SVM)',
    'knn': 'K-Nearest Neighbors (KNN)'
}
selected_model_key = st.sidebar.selectbox(
    "Active Classifier",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x]
)

# Header Section
st.markdown("<div class='main-title'>Marine Engine Diagnostics</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>Predicting engine anomalies and fault classifications using 6 ML models | Active: <strong>{model_options[selected_model_key]}</strong></div>", unsafe_allow_html=True)

# Check if model weights exist
missing_weights = not models or len(models) < 6 or scaler is None or metrics_payload is None
if missing_weights:
    st.warning("⚠️ Some model weight files (`.pkl`) or performance metrics were not detected in the workspace.")
    st.info("You can trigger the training script directly below to train all 6 classifiers and generate evaluation metrics.")
    if st.button("🚀 Train & Generate All Model Weights"):
        trigger_training()
    st.stop()

# Set current active model
model = models[selected_model_key]

# Load stats
stats_dict = get_dataset_stats()

# ----------------- TAB 0: PROJECT INFO & DOCUMENTATION -----------------
if app_mode == "📖 Project Info & Documentation":
    st.markdown("### 📖 Project Information & Documentation")
    st.write("Welcome to the **Marine Engine Health & Fault Diagnostics Hub**. This system utilizes a machine learning classifier to interpret engine sensor telemetry and identify operational anomalies.")
    
    col_docs_1, col_docs_2 = st.columns([3, 2])
    
    with col_docs_1:
        st.markdown("#### 🎯 Project Objective & Scope")
        st.write("""
        This project implements a predictive maintenance framework for commercial shipping vessel engines. 
        By continuously feeding telemetry data from 18 physical sensor checkpoints into a machine learning model, the crew can catch critical faults (like low cylinder compression, fuel delivery problems, lubrication thermal runaways, or excessive vibration) before they escalate into catastrophic mechanical failures.
        """)
        
        st.markdown("#### 🧠 Multi-Model Diagnostics Pipeline")
        st.write("""
        - **Data Scaler**: Inputs are standardized using a pre-fit `StandardScaler` (mean=0, variance=1) before being processed by any model.
        - **Algorithms Integrated**:
          1. **Logistic Regression** (L2 Regularized) - Serves as an interpretable statistical baseline.
          2. **Random Forest Classifier** - Robust bagging ensemble model.
          3. **XGBoost Classifier** - High-accuracy gradient boosted trees classifier.
          4. **Decision Tree Classifier** - Quick, rule-based hierarchical classifier.
          5. **Support Vector Machine (SVM)** - RBF kernel distance-based classifier.
          6. **K-Nearest Neighbors (KNN)** - Instance-based local classifier (k=5).
        """)
        
    with col_docs_2:
        st.markdown("#### 🚢 Engine System Diagram")
        st.info("📊 **Sensors Monitored**: RPM, Load, Fuel Flow, Manifold Air Pressure, Ambient Temperature, Lube Oil Temperature, Lube Oil Pressure, 3-Axis Vibration (X/Y/Z), and individual compression & exhaust temperatures for all 4 cylinders.")
        
    st.write("")
    
    # Sensors table
    st.markdown("#### ⚙️ Feature Matrix: Engine Telemetry Sensors (18 Features)")
    st.write("These variables represent the physical indicators measured continuously from the engine:")
    
    features_table_data = [
        {"Sensor Variable": "Shaft_RPM", "Unit": "RPM", "Category": "Mechanical Operation", "Description": "Rotational speed of the primary engine drive shaft. Used as baseline speed for fuel/load diagnostics."},
        {"Sensor Variable": "Engine_Load", "Unit": "%", "Category": "Mechanical Operation", "Description": "Current load demands placed on the engine relative to its maximum capacity."},
        {"Sensor Variable": "Fuel_Flow", "Unit": "L/h", "Category": "Fuel Delivery", "Description": "Rate of fuel supplied to the combustion chambers. Key indicator of fuel feed problems."},
        {"Sensor Variable": "Air_Pressure", "Unit": "bar", "Category": "Intake & Combustion", "Description": "Manifold boost air pressure delivered to cylinders. Used to detect turbocharger anomalies."},
        {"Sensor Variable": "Ambient_Temp", "Unit": "°C", "Category": "Environmental", "Description": "Environmental air temperature surrounding the engine compartment."},
        {"Sensor Variable": "Oil_Temp", "Unit": "°C", "Category": "Lubrication System", "Description": "Temperature of the lube oil in the sump. Increases heavily during high-friction anomaly states."},
        {"Sensor Variable": "Oil_Pressure", "Unit": "bar", "Category": "Lubrication System", "Description": "Feed oil pressure delivered to the crankshaft journals and bearings."},
        {"Sensor Variable": "Vibration_X", "Unit": "g", "Category": "Mechanical Vibration", "Description": "Engine block vibration along the lateral radial axis."},
        {"Sensor Variable": "Vibration_Y", "Unit": "g", "Category": "Mechanical Vibration", "Description": "Engine block vibration along the vertical radial axis."},
        {"Sensor Variable": "Vibration_Z", "Unit": "g", "Category": "Mechanical Vibration", "Description": "Engine block vibration along the longitudinal axial axis."},
        {"Sensor Variable": "Cylinder1_Pressure", "Unit": "bar", "Category": "Cylinder Compression", "Description": "Peak compression pressure inside Cylinder 1 during combustion."},
        {"Sensor Variable": "Cylinder1_Exhaust_Temp", "Unit": "°C", "Category": "Cylinder Exhaust", "Description": "Temperature of the exhaust gases leaving Cylinder 1."},
        {"Sensor Variable": "Cylinder2_Pressure", "Unit": "bar", "Category": "Cylinder Compression", "Description": "Peak compression pressure inside Cylinder 2 during combustion."},
        {"Sensor Variable": "Cylinder2_Exhaust_Temp", "Unit": "°C", "Category": "Cylinder Exhaust", "Description": "Temperature of the exhaust gases leaving Cylinder 2."},
        {"Sensor Variable": "Cylinder3_Pressure", "Unit": "bar", "Category": "Cylinder Compression", "Description": "Peak compression pressure inside Cylinder 3 during combustion."},
        {"Sensor Variable": "Cylinder3_Exhaust_Temp", "Unit": "°C", "Category": "Cylinder Exhaust", "Description": "Temperature of the exhaust gases leaving Cylinder 3."},
        {"Sensor Variable": "Cylinder4_Pressure", "Unit": "bar", "Category": "Cylinder Compression", "Description": "Peak compression pressure inside Cylinder 4 during combustion."},
        {"Sensor Variable": "Cylinder4_Exhaust_Temp", "Unit": "°C", "Category": "Cylinder Exhaust", "Description": "Temperature of the exhaust gases leaving Cylinder 4."}
    ]
    st.table(pd.DataFrame(features_table_data))
    
    st.write("")
    
    # Faults table
    st.markdown("#### 🚨 Target Matrix: Diagnostic Classifications & Associated Symptoms (8 Classes)")
    st.write("Based on regression coefficients and dataset statistics, the 8 classes represent the following operational conditions:")
    
    faults_table_data = []
    for label, info in FAULT_CLASSES.items():
        faults_table_data.append({
            "Label ID": label,
            "Diagnostic Name": info["name"],
            "Severity": info["severity"],
            "Description": info["desc"]
        })
    st.table(pd.DataFrame(faults_table_data))

# ----------------- TAB 1: MODEL COMPARISON & ANALYTICS -----------------
elif app_mode == "📊 Model Comparison & Analytics":
    st.markdown("### 📊 Model Comparison & Performance Analytics")
    
    if metrics_payload and 'metrics' in metrics_payload:
        metrics_dict = metrics_payload['metrics']
        
        # Calculate best model and statistics
        best_acc_model = max(metrics_dict.items(), key=lambda x: x[1]['accuracy'])
        fastest_inf_model = min(metrics_dict.items(), key=lambda x: x[1]['inference_time_ms_per_sample'])
        
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Top Model (Accuracy)</div>
                <div class='metric-value'>{best_acc_model[1]['name']}</div>
                <div class='metric-desc'>Validation Accuracy: {best_acc_model[1]['accuracy']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Fastest Inference</div>
                <div class='metric-value'>{fastest_inf_model[1]['name']}</div>
                <div class='metric-desc'>Latency: {fastest_inf_model[1]['inference_time_ms_per_sample']:.4f} ms/sample</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-label'>Total Classifiers</div>
                <div class='metric-value'>6 Models</div>
                <div class='metric-desc'>Trained & validated on dataset</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class='metric-card'>
                <div class='metric-label'>Telemetry Sensors</div>
                <div class='metric-value'>18 Physical Features</div>
                <div class='metric-desc'>Scaled input values</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # Overall Performance Chart (Accuracy, Precision, Recall, F1)
        st.markdown("#### 🔬 Model Metric Comparison")
        st.write("Comparison of key metrics across the 6 classifiers evaluated on the test set split.")
        
        comparison_data = []
        for key, m_data in metrics_dict.items():
            rep = m_data['report']
            # Get macro averages
            precision = rep['macro avg']['precision']
            recall = rep['macro avg']['recall']
            f1 = rep['macro avg']['f1-score']
            accuracy = m_data['accuracy']
            
            comparison_data.append({
                'Model': m_data['name'],
                'Accuracy': accuracy,
                'Precision (Macro)': precision,
                'Recall (Macro)': recall,
                'F1-Score (Macro)': f1,
                'Inference Speed (ms)': m_data['inference_time_ms_per_sample'],
                'Train Time (s)': m_data['train_time_seconds']
            })
            
        df_compare = pd.DataFrame(comparison_data)
        
        # Plotly chart: multi bar chart
        df_melted = df_compare.melt(id_vars='Model', value_vars=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
                                     var_name='Metric', value_name='Score')
        
        fig_metrics = px.bar(
            df_melted,
            x='Model',
            y='Score',
            color='Metric',
            barmode='group',
            color_discrete_sequence=['#60a5fa', '#10B981', '#F59E0B', '#EF4444'],
            labels={'Score': 'Score Value'},
            height=400
        )
        fig_metrics.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            yaxis=dict(range=[0, 1.05]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        st.markdown("##### 📋 Model Performance Metrics Comparison Table")
        st.dataframe(
            df_compare.style.format({
                'Accuracy': '{:.2%}',
                'Precision (Macro)': '{:.2%}',
                'Recall (Macro)': '{:.2%}',
                'F1-Score (Macro)': '{:.2%}',
                'Inference Speed (ms)': '{:.4f} ms',
                'Train Time (s)': '{:.2f} s'
            }).background_gradient(cmap='Blues', subset=['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']),
            use_container_width=True
        )
        
        st.write("")
        
        col_speed_l, col_speed_r = st.columns(2)
        with col_speed_l:
            st.markdown("#### ⚡ Inference Latency Comparison")
            st.write("Average prediction latency in milliseconds per sample (lower is faster).")
            fig_speed = px.bar(
                df_compare,
                x='Model',
                y='Inference Speed (ms)',
                color='Inference Speed (ms)',
                color_continuous_scale='Reds',
                labels={'Inference Speed (ms)': 'Latency (ms)'},
                height=300
            )
            fig_speed.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'),
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=10, b=20)
            )
            st.plotly_chart(fig_speed, use_container_width=True)
            
        with col_speed_r:
            st.markdown("#### ⏳ Training Time Comparison")
            st.write("Total training time in seconds (lower is faster).")
            fig_train = px.bar(
                df_compare,
                x='Model',
                y='Train Time (s)',
                color='Train Time (s)',
                color_continuous_scale='Purples',
                labels={'Train Time (s)': 'Time (seconds)'},
                height=300
            )
            fig_train.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'),
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=10, b=20)
            )
            st.plotly_chart(fig_train, use_container_width=True)
            
        st.write("")
        st.markdown("---")
        
        # Model-Specific Analytics Selector
        st.markdown("#### 🔍 Model Diagnostics Details")
        st.write("Select any trained model to examine its details (confusion matrix, classification report, feature importances/coefficients).")
        
        selected_diag_key = st.selectbox(
            "Select Model for Deep Analysis",
            options=list(metrics_dict.keys()),
            format_func=lambda x: metrics_dict[x]['name']
        )
        
        selected_metric = metrics_dict[selected_diag_key]
        
        col_det_l, col_det_r = st.columns(2)
        with col_det_l:
            st.markdown(f"##### 🎛️ Confusion Matrix: {selected_metric['name']}")
            cm = np.array(selected_metric['cm'])
            class_labels = [f"Class {i}" for i in range(len(cm))]
            
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted Class Label", y="Ground Truth Label", color="Recordings Count"),
                x=class_labels,
                y=class_labels,
                color_continuous_scale="Viridis",
                text_auto=True,
                aspect="auto"
            )
            fig_cm.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_det_r:
            st.markdown(f"##### 📋 Classification Report: {selected_metric['name']}")
            report_df = pd.DataFrame(selected_metric['report']).transpose().iloc[:-3] # exclude accuracy, macro avg, weighted avg
            report_df.index = [f"Class {i}: {FAULT_CLASSES[int(float(i))]['name']}" for i in report_df.index]
            
            report_df = report_df.rename(columns={
                'precision': 'Precision',
                'recall': 'Recall',
                'f1-score': 'F1-Score',
                'support': 'Total Instances'
            })
            st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['Precision', 'Recall', 'F1-Score']), use_container_width=True)
            
        st.write("")
        
        # Feature Importance / Coefficients Section
        st.markdown(f"##### 📊 Feature Relevance: {selected_metric['name']}")
        
        model_obj = models.get(selected_diag_key)
        feature_names = metrics_payload.get('feature_names', [])
        
        if model_obj is not None:
            if hasattr(model_obj, 'feature_importances_'):
                importances = model_obj.feature_importances_
                df_imp = pd.DataFrame({
                    'Sensor Variable': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                fig_imp = px.bar(
                    df_imp,
                    x='Importance',
                    y='Sensor Variable',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Blues',
                    title=f'{selected_metric["name"]} - Feature Importances (Sum = 1.0)',
                    height=450
                )
                fig_imp.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c9d1d9'),
                    coloraxis_showscale=False,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
            elif hasattr(model_obj, 'coef_'):
                # For Logistic Regression
                coefs = model_obj.coef_
                class_names = [f"Class {i}: {FAULT_CLASSES[i]['name']}" for i in range(len(coefs))]
                
                fig_coef = px.imshow(
                    coefs,
                    labels=dict(x="Engine Sensor Variable", y="Predicted Diagnostics Class", color="Coefficient Weight"),
                    x=feature_names,
                    y=class_names,
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    aspect="auto",
                    title="Logistic Regression - Diagnostic Coefficients Map (Red=Positive, Blue=Negative)"
                )
                fig_coef.update_layout(
                    xaxis_tickangle=-45,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#c9d1d9'),
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig_coef, use_container_width=True)
            else:
                st.info("ℹ️ Distance-based classifiers like Support Vector Machines (RBF Kernel) and K-Nearest Neighbors do not project direct linear coefficients or tree-based feature importances.")
        else:
            st.warning("Could not load the model weights object to extract feature importances.")
            
    else:
        st.info("Performance analytics metrics are not available. Please retrain models to generate them.")

# ----------------- TAB 2: SINGLE ENGINE DIAGNOSTICS -----------------
elif app_mode == "🔍 Single Engine Diagnostics":
    st.markdown("### 🔍 Real-Time Sensor Diagnosis Station")
    st.write(f"Modify physical sensor parameters below. The values will be scaled and evaluated using the active model: **{model_options[selected_model_key]}**.")

    # Initialize slider state values in session_state if not present
    for col, limits in stats_dict.items():
        key = f"slider_{col}"
        if key not in st.session_state:
            st.session_state[key] = limits[2]

    # Create Columns for Categories
    col_inputs, col_result = st.columns([7, 5])
    
    with col_inputs:
        st.markdown("#### 🛠️ Sensor Inputs")
        
        # Group 1: Mechanical Operation
        st.markdown("<div class='feature-group-header'>⚙️ Primary Engine Operation Parameters</div>", unsafe_allow_html=True)
        col_g1_1, col_g1_2 = st.columns(2)
        with col_g1_1:
            limits = stats_dict.get('Shaft_RPM', (750.0, 1150.0, 960.0))
            shaft_rpm = st.slider("Shaft RPM (rotations/min)", min_value=limits[0], max_value=limits[1], key='slider_Shaft_RPM', step=1.0)
            
            limits = stats_dict.get('Engine_Load', (25.0, 110.0, 75.0))
            engine_load = st.slider("Engine Load (%)", min_value=limits[0], max_value=limits[1], key='slider_Engine_Load', step=0.5)
        with col_g1_2:
            limits = stats_dict.get('Fuel_Flow', (60.0, 190.0, 130.0))
            fuel_flow = st.slider("Fuel Flow Rate (L/h)", min_value=limits[0], max_value=limits[1], key='slider_Fuel_Flow', step=0.5)
            
            limits = stats_dict.get('Air_Pressure', (0.3, 1.6, 1.15))
            air_pressure = st.slider("Air Pressure (bar)", min_value=limits[0], max_value=limits[1], key='slider_Air_Pressure', step=0.01)

        # Group 2: Lubrication & Thermal Condition
        st.markdown("<div class='feature-group-header'>🌡️ Lubrication & Thermal Diagnostics</div>", unsafe_allow_html=True)
        col_g2_1, col_g2_2 = st.columns(2)
        with col_g2_1:
            limits = stats_dict.get('Ambient_Temp', (15.0, 40.0, 27.0))
            ambient_temp = st.slider("Ambient Temperature (°C)", min_value=limits[0], max_value=limits[1], key='slider_Ambient_Temp', step=0.1)
            
            limits = stats_dict.get('Oil_Temp', (60.0, 115.0, 78.0))
            oil_temp = st.slider("Oil Temperature (°C)", min_value=limits[0], max_value=limits[1], key='slider_Oil_Temp', step=0.1)
        with col_g2_2:
            limits = stats_dict.get('Oil_Pressure', (0.4, 5.2, 3.4))
            oil_pressure = st.slider("Oil Pressure (bar)", min_value=limits[0], max_value=limits[1], key='slider_Oil_Pressure', step=0.05)

        # Group 3: Vibration Channels
        st.markdown("<div class='feature-group-header'>📳 Vibration Sensors</div>", unsafe_allow_html=True)
        col_g3_1, col_g3_2, col_g3_3 = st.columns(3)
        with col_g3_1:
            limits = stats_dict.get('Vibration_X', (0.0, 0.5, 0.06))
            vib_x = st.slider("Vibration X (g-force)", min_value=limits[0], max_value=limits[1], key='slider_Vibration_X', step=0.005)
        with col_g3_2:
            limits = stats_dict.get('Vibration_Y', (0.0, 0.5, 0.05))
            vib_y = st.slider("Vibration Y (g-force)", min_value=limits[0], max_value=limits[1], key='slider_Vibration_Y', step=0.005)
        with col_g3_3:
            limits = stats_dict.get('Vibration_Z', (0.0, 0.6, 0.07))
            vib_z = st.slider("Vibration Z (g-force)", min_value=limits[0], max_value=limits[1], key='slider_Vibration_Z', step=0.005)

        # Group 4: Combustion Chambers
        st.markdown("<div class='feature-group-header'>🔥 Combustion Cylinder Pressures & Exhaust Temperatures</div>", unsafe_allow_html=True)
        
        with st.expander("Cylinder Compression Pressures"):
            col_c_p1, col_c_p2 = st.columns(2)
            with col_c_p1:
                limits = stats_dict.get('Cylinder1_Pressure', (85.0, 190.0, 145.0))
                cyl1_p = st.slider("Cylinder 1 Pressure (bar)", min_value=limits[0], max_value=limits[1], key='slider_Cylinder1_Pressure', step=0.5)
                
                limits = stats_dict.get('Cylinder2_Pressure', (90.0, 190.0, 145.0))
                cyl2_p = st.slider("Cylinder 2 Pressure (bar)", min_value=limits[0], max_value=limits[1], key='slider_Cylinder2_Pressure', step=0.5)
            with col_c_p2:
                limits = stats_dict.get('Cylinder3_Pressure', (85.0, 190.0, 145.0))
                cyl3_p = st.slider("Cylinder 3 Pressure (bar)", min_value=limits[0], max_value=limits[1], key='slider_Cylinder3_Pressure', step=0.5)
                
                limits = stats_dict.get('Cylinder4_Pressure', (85.0, 190.0, 145.0))
                cyl4_p = st.slider("Cylinder 4 Pressure (bar)", min_value=limits[0], max_value=limits[1], key='slider_Cylinder4_Pressure', step=0.5)

        with st.expander("Cylinder Exhaust Gas Temperatures"):
            col_c_t1, col_c_t2 = st.columns(2)
            with col_c_t1:
                limits = stats_dict.get('Cylinder1_Exhaust_Temp', (290.0, 620.0, 420.0))
                cyl1_t = st.slider("Cylinder 1 Exhaust (°C)", min_value=limits[0], max_value=limits[1], key='slider_Cylinder1_Exhaust_Temp', step=1.0)
                
                limits = stats_dict.get('Cylinder2_Exhaust_Temp', (310.0, 600.0, 420.0))
                cyl2_t = st.slider("Cylinder 2 Exhaust (°C)", min_value=limits[0], max_value=limits[1], key='slider_Cylinder2_Exhaust_Temp', step=1.0)
            with col_c_t2:
                limits = stats_dict.get('Cylinder3_Exhaust_Temp', (300.0, 610.0, 420.0))
                cyl3_t = st.slider("Cylinder 3 Exhaust (°C)", min_value=limits[0], max_value=limits[1], key='slider_Cylinder3_Exhaust_Temp', step=1.0)
                
                limits = stats_dict.get('Cylinder4_Exhaust_Temp', (310.0, 620.0, 420.0))
                cyl4_t = st.slider("Cylinder 4 Exhaust (°C)", min_value=limits[0], max_value=limits[1], key='slider_Cylinder4_Exhaust_Temp', step=1.0)

    with col_result:
        st.markdown("#### 🚨 Diagnosis Result")
        
        # Consolidate feature array
        input_data = np.array([[
            shaft_rpm, engine_load, fuel_flow, air_pressure, ambient_temp, oil_temp, oil_pressure,
            vib_x, vib_y, vib_z, cyl1_p, cyl1_t, cyl2_p, cyl2_t, cyl3_p, cyl3_t, cyl4_p, cyl4_t
        ]])
        
        # Scale inputs
        input_scaled = scaler.transform(input_data)
        
        # Run prediction
        pred_class = int(model.predict(input_scaled)[0])
        pred_probs = model.predict_proba(input_scaled)[0]
        confidence = pred_probs[pred_class]
        
        class_info = FAULT_CLASSES[pred_class]
        severity_color = class_info["color"]
        severity_label = class_info["severity"]
        
        glow_shadow = f"box-shadow: 0 0 25px {severity_color}35;"
        
        # Diagnosis Status box
        st.markdown(f"""
        <div class='diag-card' style='border-left: 8px solid {severity_color}; {glow_shadow}'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-size: 0.9rem; color: #9ca3af; font-weight: 700; text-transform: uppercase;'>Engine Status ({model_options[selected_model_key]})</span>
                <span style='background-color: {severity_color}25; color: {severity_color}; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700; border: 1px solid {severity_color};'>{severity_label}</span>
            </div>
            <div style='font-size: 1.5rem; font-weight: 800; color: #ffffff; margin-top: 0.6rem;'>
                {class_info["name"]}
            </div>
            <div style='font-size: 0.95rem; color: #9ca3af; margin-top: 0.6rem; line-height: 1.4;'>
                {class_info["desc"]}
            </div>
            <div style='margin-top: 1.2rem; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 0.6rem; font-size: 0.85rem; color: #9ca3af;'>
                Confidence Score: <strong style='color: #ffffff; font-size: 1rem;'>{confidence:.2%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilities Bar Chart
        st.write("")
        st.markdown("##### 📊 Probability Distribution")
        
        probs_df = pd.DataFrame({
            'Diagnosis': [FAULT_CLASSES[i]['name'] for i in range(len(pred_probs))],
            'Probability (%)': pred_probs * 100
        })
        
        fig_probs = px.bar(
            probs_df,
            x='Probability (%)',
            y='Diagnosis',
            orientation='h',
            text='Probability (%)',
            color='Probability (%)',
            color_continuous_scale='Blues',
            range_x=[0, 100]
        )
        fig_probs.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_probs.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c9d1d9'),
            height=320,
            xaxis=dict(showgrid=False),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=50, t=10, b=10)
        )
        st.plotly_chart(fig_probs, use_container_width=True)

        # Multi-model real-time check
        st.write("")
        with st.expander("🤖 Real-Time Multi-Model Comparison"):
            st.write("Compare diagnosis results from all 6 ML classifiers on current slider parameters:")
            
            multi_preds = []
            for m_key, m_obj in models.items():
                m_pred = int(m_obj.predict(input_scaled)[0])
                m_probs = m_obj.predict_proba(input_scaled)[0]
                m_conf = m_probs[m_pred]
                m_info = FAULT_CLASSES[m_pred]
                
                multi_preds.append({
                    'Classifier Model': model_options[m_key],
                    'Predicted Status': m_info['name'],
                    'Severity': m_info['severity'],
                    'Confidence Score': f"{m_conf:.2%}"
                })
                
            st.dataframe(pd.DataFrame(multi_preds), use_container_width=True)

        # Quick Presets Sandbox
        st.write("")
        st.markdown("##### 💡 Anomaly Presets Sandbox")
        st.write("Instantly load sensor value patterns to simulate specific engine conditions:")
        
        # Display the message if any is set in session_state
        if 'preset_message' in st.session_state:
            msg_type, text = st.session_state['preset_message']
            if msg_type == "healthy":
                st.info(text)
            else:
                st.warning(text)
                
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("🟢 Normal Operation preset", on_click=load_preset, args=("healthy",), use_container_width=True)
            st.button("🔴 Fuel Delivery Anomaly preset", on_click=load_preset, args=("fuel",), use_container_width=True)
            st.button("🔴 Low Compression preset", on_click=load_preset, args=("compression",), use_container_width=True)
            st.button("🔴 Exhaust Heat Anomaly preset", on_click=load_preset, args=("exhaust",), use_container_width=True)
        with col_btn2:
            st.button("🔴 Radial Vibration preset", on_click=load_preset, args=("vibration",), use_container_width=True)
            st.button("🔴 Lubrication Thermal preset", on_click=load_preset, args=("lube_thermal",), use_container_width=True)
            st.button("🔴 Turbocharger Fault preset", on_click=load_preset, args=("turbo",), use_container_width=True)
            st.button("🔴 Lube Pressure & Axial Vib preset", on_click=load_preset, args=("lube_axial",), use_container_width=True)

# ----------------- TAB 3: BATCH FILE DIAGNOSTICS -----------------
elif app_mode == "📁 Batch File Diagnostics":
    st.markdown("### 📁 Batch Processing Station")
    st.write(f"Upload a CSV file containing multiple engine recordings. Predictions will be processed using: **{model_options[selected_model_key]}**.")

    uploaded_file = st.file_uploader("Upload Engine Sensor CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            expected_features = [
                'Shaft_RPM', 'Engine_Load', 'Fuel_Flow', 'Air_Pressure', 'Ambient_Temp', 'Oil_Temp', 'Oil_Pressure',
                'Vibration_X', 'Vibration_Y', 'Vibration_Z', 'Cylinder1_Pressure', 'Cylinder1_Exhaust_Temp',
                'Cylinder2_Pressure', 'Cylinder2_Exhaust_Temp', 'Cylinder3_Pressure', 'Cylinder3_Exhaust_Temp',
                'Cylinder4_Pressure', 'Cylinder4_Exhaust_Temp'
            ]
            
            missing_cols = [col for col in expected_features if col not in df_upload.columns]
            if len(missing_cols) > 0:
                st.error(f"Failed to process CSV. The file is missing the following required sensor columns:\n`{missing_cols}`")
            else:
                with st.spinner("Processing batch predictions..."):
                    # Process predictions
                    x_batch = df_upload[expected_features].values
                    x_batch_scaled = scaler.transform(x_batch)
                    
                    # Selected model predict
                    preds = model.predict(x_batch_scaled)
                    probs = model.predict_proba(x_batch_scaled)
                    max_probs = np.max(probs, axis=1)
                    
                    # Add outputs to dataframe
                    df_upload['Predicted_Fault_Label'] = preds
                    df_upload['Diagnosis_Name'] = [FAULT_CLASSES[p]['name'] for p in preds]
                    df_upload['Diagnostic_Severity'] = [FAULT_CLASSES[p]['severity'] for p in preds]
                    df_upload['Confidence_Score'] = max_probs
                    
                    # Run predictions for all models to show consensus
                    consensus_cols = {}
                    for m_key, m_obj in models.items():
                        m_preds = m_obj.predict(x_batch_scaled)
                        consensus_cols[model_options[m_key]] = m_preds
                        
                    df_consensus = pd.DataFrame(consensus_cols)
                    
                    # Calculate row-wise consensus: fraction of models that agree with the majority prediction
                    majority_preds = df_consensus.mode(axis=1).iloc[:, 0].values
                    agreement_counts = (df_consensus.values == majority_preds[:, None]).sum(axis=1)
                    avg_agreement_pct = agreement_counts.mean() / 6.0
                    
                    # Metrics row
                    total_records = len(df_upload)
                    anomaly_count = len(df_upload[df_upload['Predicted_Fault_Label'] > 0])
                    healthy_count = total_records - anomaly_count
                    anomaly_pct = anomaly_count / total_records if total_records > 0 else 0
                    
                    st.write("")
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Total Records Processed</div>
                            <div class='metric-value'>{total_records:,}</div>
                            <div class='metric-desc'>Rows analyzed in CSV</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Normal Readings</div>
                            <div class='metric-value' style='color: #10B981;'>{healthy_count:,}</div>
                            <div class='metric-desc'>Nominal engine states</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m3:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Anomaly Detections</div>
                            <div class='metric-value' style='color: #EF4444;'>{anomaly_count:,}</div>
                            <div class='metric-desc'>Inspections required</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m4:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Consensus Agreement</div>
                            <div class='metric-value'>{avg_agreement_pct:.2%}</div>
                            <div class='metric-desc'>Classifier consensus index</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Multi-Model Consensus Breakdown
                    st.write("")
                    st.markdown("#### 🤖 Classifier Consensus Breakdown")
                    st.write("Compare predictions and anomaly discovery distributions across all 6 models on this batch file:")
                    
                    model_breakdown = []
                    for m_key, m_obj in models.items():
                        m_preds = consensus_cols[model_options[m_key]]
                        m_anoms = (m_preds > 0).sum()
                        model_breakdown.append({
                            'Model Classifier': model_options[m_key],
                            'Accuracy (on Test Set)': f"{metrics_payload['metrics'][m_key]['accuracy']:.2%}",
                            'Nominal Count': len(m_preds) - m_anoms,
                            'Anomaly Count': m_anoms,
                            'Anomaly Percentage': f"{m_anoms / len(m_preds):.2%}"
                        })
                    st.table(pd.DataFrame(model_breakdown))
                    
                    # Visualizations
                    st.markdown("#### 📈 Batch Metrics & Trends")
                    
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        # Pie chart of predictions
                        dist_df = df_upload['Diagnosis_Name'].value_counts().reset_index()
                        dist_df.columns = ['Diagnosis', 'Count']
                        
                        fig_pie = px.pie(
                            dist_df,
                            values='Count',
                            names='Diagnosis',
                            title=f'Diagnostics Distribution ({model_options[selected_model_key]})',
                            hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig_pie.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#c9d1d9'),
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                        
                    with col_v2:
                        # Shaft RPM vs Fuel Flow colored by predicted fault
                        fig_scatter = px.scatter(
                            df_upload,
                            x='Shaft_RPM',
                            y='Fuel_Flow',
                            color='Diagnosis_Name',
                            title='Operating Window: Shaft RPM vs Fuel Flow',
                            opacity=0.7,
                            labels={'Diagnosis_Name': 'Diagnostics'},
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        fig_scatter.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#c9d1d9'),
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                    # Preview & download
                    st.markdown("#### 📄 Diagnostics Output Preview")
                    st.dataframe(df_upload.head(100), use_container_width=True)
                    
                    # Convert to csv
                    csv_data = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Complete Diagnostics CSV File",
                        data=csv_data,
                        file_name="diagnosed_marine_engine_records.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
    else:
        st.info("💡 Upload an engine log CSV file to begin. The file must match the features from the training dataset.")
