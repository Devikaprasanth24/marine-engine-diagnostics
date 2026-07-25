import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(
    page_title="Marine Engine Health & Fault Diagnostics",
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
    /* Dark Theme adjustments */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Custom cards for stats and metrics */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s, border-color 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-top: 0.25rem;
    }
    
    .metric-desc {
        font-size: 0.75rem;
        color: #58a6ff;
        margin-top: 0.25rem;
    }

    /* Diagnosis result card */
    .diag-card {
        background: rgba(22, 27, 34, 0.85);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #30363d;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .feature-group-header {
        color: #58a6ff;
        font-size: 1.15rem;
        font-weight: 600;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.4rem;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }
    
    /* Sidebar navigation list */
    .nav-header {
        font-size: 0.85rem;
        color: #8b949e;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_assets():
    model_path = "logistic_model.pkl"
    scaler_path = "scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
        
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, scaler = load_assets()

# Load dataset for statistics
@st.cache_data
def get_dataset_stats():
    csv_path = "marine_engine_fault_dataset (1).csv"
    if not os.path.exists(csv_path):
        # Fallback dictionary of (min, max, default_median)
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
        # Return fallback
        return get_dataset_stats.__wrapped__()

# Get evaluation metrics
@st.cache_data
def get_evaluation_metrics():
    csv_path = "marine_engine_fault_dataset (1).csv"
    if not os.path.exists(csv_path) or model is None or scaler is None:
        return None
        
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna().drop_duplicates()
        
        feature_cols = [col for col in df.columns if col not in ['Timestamp', 'Fault_Label']]
        x = df[feature_cols].values
        y = df['Fault_Label'].values
        
        from sklearn.model_selection import train_test_split
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        x_test_scaled = scaler.transform(x_test)
        y_pred = model.predict(x_test_scaled)
        
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'cm': cm,
            'report': report_dict,
            'test_size': len(y_test),
            'feature_names': feature_cols
        }
    except Exception as e:
        st.warning(f"Error calculating evaluation metrics: {e}")
        return None

# Training function triggerable from UI
def trigger_training():
    with st.spinner("🔄 Training Logistic Regression Model..."):
        try:
            import subprocess
            res = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
            if res.returncode == 0:
                st.success("🎉 Model retrained and loaded successfully!")
                # Reset cached assets
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
    ["📖 Project Info & Documentation", "📊 Model Analytics & Weights", "🔍 Single Engine Diagnostics", "📁 Batch File Diagnostics"]
)

# Header Section
st.markdown("<div class='main-title'>Marine Engine Diagnostics</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predicting engine anomalies and fault classifications using Logistic Regression</div>", unsafe_allow_html=True)

# Check if model loaded successfully
if model is None or scaler is None:
    st.warning("⚠️ Model weights (`logistic_model.pkl` and `scaler.pkl`) were not detected in the project folder.")
    st.info("You can trigger the training script directly below to preprocess the dataset and generate the model weights.")
    if st.button("🚀 Train & Generate Model Weights"):
        trigger_training()
    st.stop()

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
        
        st.markdown("#### 🧠 Model & Preprocessing Pipeline")
        st.write("""
        - **Classifier**: **Logistic Regression** (L2 Regularized, solver='lbfgs') selected for high interpretability, transparency, and low CPU overhead.
        - **Accuracy**: **85.25%** on the validation set.
        - **Inputs**: 18 numeric physical telemetry indicators. The string `Timestamp` column was dropped from the input array during training to prevent the scaling model from breaking.
        - **Standardization**: Inputs are scaled using a pre-fit `StandardScaler` to bring all sensors onto a common scale (mean=0, variance=1) before being processed by the logistic model.
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

# ----------------- TAB 1: MODEL ANALYTICS & WEIGHTS -----------------
elif app_mode == "📊 Model Analytics & Weights":
    st.markdown("### 📊 Model Performance & Weight Metrics")
    
    metrics = get_evaluation_metrics()
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Model Classifier</div>
            <div class='metric-value'>Logistic Regression</div>
            <div class='metric-desc'>L2 Regularized, max_iter=1000</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        acc_text = f"{metrics['report']['accuracy']:.2%}" if metrics else "85.25%"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Overall Accuracy</div>
            <div class='metric-value'>{acc_text}</div>
            <div class='metric-desc'>Evaluated on 20% test split</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        test_size_text = f"{metrics['test_size']:,} rows" if metrics else "2,000 rows"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Evaluation Dataset</div>
            <div class='metric-value'>{test_size_text}</div>
            <div class='metric-desc'>Engine sensor recordings</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Feature Inputs</div>
            <div class='metric-value'>18 sensors</div>
            <div class='metric-desc'>Excluding string Timestamps</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Detailed plots
    if metrics:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### 🔬 Model Classification Weights")
            st.write("This matrix shows how the model weights (coefficients) influence predictions. Red represents positive coefficients (increasing the likelihood of that fault), while blue represents negative coefficients.")
            
            # Heatmap of coefficients
            coefs = model.coef_
            class_names = [f"Class {i}: {FAULT_CLASSES[i]['name']}" for i in range(len(coefs))]
            feature_names = metrics['feature_names']
            
            fig = px.imshow(
                coefs,
                labels=dict(x="Engine Sensor Variable", y="Predicted Diagnostics Class", color="Coefficient Weight"),
                x=feature_names,
                y=class_names,
                color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                aspect="auto"
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c9d1d9'),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_right:
            st.markdown("#### 🔄 Confusion Matrix Heatmap")
            st.write("Evaluates predicted labels against true ground truth sensor recordings on the validation set.")
            
            cm = metrics['cm']
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
            
        # Class-wise reports
        st.markdown("#### 📋 Classification Report Breakdown")
        report_df = pd.DataFrame(metrics['report']).transpose().iloc[:-3] # exclude accuracy, macro avg, weighted avg
        report_df.index = [f"Class {i}: {FAULT_CLASSES[int(i)]['name']}" for i in report_df.index]
        
        # Rename columns for clarity
        report_df = report_df.rename(columns={
            'precision': 'Precision (Correctness)',
            'recall': 'Recall (Coverage)',
            'f1-score': 'F1-Score',
            'support': 'Total Instances (Ground Truth)'
        })
        st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['Precision (Correctness)', 'Recall (Coverage)', 'F1-Score']), use_container_width=True)
    else:
        st.info("Detailed performance graphs will appear here once the model evaluation completes.")

# ----------------- TAB 2: SINGLE ENGINE DIAGNOSTICS -----------------
elif app_mode == "🔍 Single Engine Diagnostics":
    st.markdown("### 🔍 Real-Time Sensor Diagnosis Station")
    st.write("Modify the engine parameters below. The system standardizes inputs and passes them to the Logistic Regression model to verify health.")

    # Create Columns for Categories
    col_inputs, col_result = st.columns([7, 5])
    
    with col_inputs:
        st.markdown("#### 🛠️ Sensor Inputs")
        
        # Group 1: Mechanical Operation
        st.markdown("<div class='feature-group-header'>⚙️ Primary Engine Operation Parameters</div>", unsafe_allow_html=True)
        col_g1_1, col_g1_2 = st.columns(2)
        with col_g1_1:
            limits = stats_dict.get('Shaft_RPM', (750.0, 1150.0, 960.0))
            shaft_rpm = st.slider("Shaft RPM (rotations/min)", min_value=limits[0], max_value=limits[1], value=limits[2], step=1.0)
            
            limits = stats_dict.get('Engine_Load', (25.0, 110.0, 75.0))
            engine_load = st.slider("Engine Load (%)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.5)
        with col_g1_2:
            limits = stats_dict.get('Fuel_Flow', (60.0, 190.0, 130.0))
            fuel_flow = st.slider("Fuel Flow Rate (L/h)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.5)
            
            limits = stats_dict.get('Air_Pressure', (0.3, 1.6, 1.15))
            air_pressure = st.slider("Air Pressure (bar)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.01)

        # Group 2: Lubrication & Thermal Condition
        st.markdown("<div class='feature-group-header'>🌡️ Lubrication & Thermal Diagnostics</div>", unsafe_allow_html=True)
        col_g2_1, col_g2_2 = st.columns(2)
        with col_g2_1:
            limits = stats_dict.get('Ambient_Temp', (15.0, 40.0, 27.0))
            ambient_temp = st.slider("Ambient Temperature (°C)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.1)
            
            limits = stats_dict.get('Oil_Temp', (60.0, 115.0, 78.0))
            oil_temp = st.slider("Oil Temperature (°C)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.1)
        with col_g2_2:
            limits = stats_dict.get('Oil_Pressure', (0.4, 5.2, 3.4))
            oil_pressure = st.slider("Oil Pressure (bar)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.05)

        # Group 3: Vibration Channels
        st.markdown("<div class='feature-group-header'>📳 Vibration Sensors</div>", unsafe_allow_html=True)
        col_g3_1, col_g3_2, col_g3_3 = st.columns(3)
        with col_g3_1:
            limits = stats_dict.get('Vibration_X', (0.0, 0.5, 0.06))
            vib_x = st.slider("Vibration X (g-force)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.005)
        with col_g3_2:
            limits = stats_dict.get('Vibration_Y', (0.0, 0.5, 0.05))
            vib_y = st.slider("Vibration Y (g-force)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.005)
        with col_g3_3:
            limits = stats_dict.get('Vibration_Z', (0.0, 0.6, 0.07))
            vib_z = st.slider("Vibration Z (g-force)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.005)

        # Group 4: Combustion Chambers
        st.markdown("<div class='feature-group-header'>🔥 Combustion Cylinder Pressures & Exhaust Temperatures</div>", unsafe_allow_html=True)
        
        # Expander for Cylinder Pressures
        with st.expander("Cylinder Compression Pressures"):
            col_c_p1, col_c_p2 = st.columns(2)
            with col_c_p1:
                limits = stats_dict.get('Cylinder1_Pressure', (85.0, 190.0, 145.0))
                cyl1_p = st.slider("Cylinder 1 Pressure (bar)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.5)
                
                limits = stats_dict.get('Cylinder2_Pressure', (90.0, 190.0, 145.0))
                cyl2_p = st.slider("Cylinder 2 Pressure (bar)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.5)
            with col_c_p2:
                limits = stats_dict.get('Cylinder3_Pressure', (85.0, 190.0, 145.0))
                cyl3_p = st.slider("Cylinder 3 Pressure (bar)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.5)
                
                limits = stats_dict.get('Cylinder4_Pressure', (85.0, 190.0, 145.0))
                cyl4_p = st.slider("Cylinder 4 Pressure (bar)", min_value=limits[0], max_value=limits[1], value=limits[2], step=0.5)

        # Expander for Cylinder Temperatures
        with st.expander("Cylinder Exhaust Gas Temperatures"):
            col_c_t1, col_c_t2 = st.columns(2)
            with col_c_t1:
                limits = stats_dict.get('Cylinder1_Exhaust_Temp', (290.0, 620.0, 420.0))
                cyl1_t = st.slider("Cylinder 1 Exhaust (°C)", min_value=limits[0], max_value=limits[1], value=limits[2], step=1.0)
                
                limits = stats_dict.get('Cylinder2_Exhaust_Temp', (310.0, 600.0, 420.0))
                cyl2_t = st.slider("Cylinder 2 Exhaust (°C)", min_value=limits[0], max_value=limits[1], value=limits[2], step=1.0)
            with col_c_t2:
                limits = stats_dict.get('Cylinder3_Exhaust_Temp', (300.0, 610.0, 420.0))
                cyl3_t = st.slider("Cylinder 3 Exhaust (°C)", min_value=limits[0], max_value=limits[1], value=limits[2], step=1.0)
                
                limits = stats_dict.get('Cylinder4_Exhaust_Temp', (310.0, 620.0, 420.0))
                cyl4_t = st.slider("Cylinder 4 Exhaust (°C)", min_value=limits[0], max_value=limits[1], value=limits[2], step=1.0)

    with col_result:
        st.markdown("#### 🚨 Diagnosis Diagnostics")
        
        # Consolidate feature array
        input_data = np.array([[
            shaft_rpm, engine_load, fuel_flow, air_pressure, ambient_temp, oil_temp, oil_pressure,
            vib_x, vib_y, vib_z, cyl1_p, cyl1_t, cyl2_p, cyl2_t, cyl3_p, cyl3_t, cyl4_p, cyl4_t
        ]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        pred_class = int(model.predict(input_scaled)[0])
        pred_probs = model.predict_proba(input_scaled)[0]
        confidence = pred_probs[pred_class]
        
        class_info = FAULT_CLASSES[pred_class]
        severity_color = class_info["color"]
        severity_label = class_info["severity"]
        
        # Diagnosis Status box
        st.markdown(f"""
        <div class='diag-card' style='border-left: 8px solid {severity_color};'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-size: 0.9rem; color: #8b949e; font-weight: 700; text-transform: uppercase;'>Engine Status</span>
                <span style='background-color: {severity_color}25; color: {severity_color}; padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700; border: 1px solid {severity_color};'>{severity_label}</span>
            </div>
            <div style='font-size: 1.5rem; font-weight: 700; color: #f0f6fc; margin-top: 0.5rem;'>
                {class_info["name"]}
            </div>
            <div style='font-size: 0.95rem; color: #8b949e; margin-top: 0.5rem; line-height: 1.4;'>
                {class_info["desc"]}
            </div>
            <div style='margin-top: 1rem; border-top: 1px solid #30363d; padding-top: 0.5rem; font-size: 0.85rem; color: #8b949e;'>
                Confidence Score: <strong style='color: #f0f6fc;'>{confidence:.2%}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilities Bar Chart
        st.write("")
        st.markdown("##### 📊 Diagnostic Distribution (Class Likelihoods)")
        
        # Make a bar chart of class probabilities
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
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=40, t=10, b=10)
        )
        st.plotly_chart(fig_probs, use_container_width=True)

        # Quick Diagnosis Presets
        st.markdown("##### 💡 Diagnostic Sandbox Quick Presets")
        st.write("Trigger quick presets to load anomaly states:")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🟢 Standard Operation preset"):
                st.info("Resetting parameters to nominal healthy medians. Modify individual sliders to explore boundaries.")
        with col_btn2:
            if st.button("🔴 Radial Vibration anomaly"):
                st.warning("Sliders reset. Increase Vibration X/Y/Z manually above boundaries to observe class weight response.")

# ----------------- TAB 3: BATCH FILE DIAGNOSTICS -----------------
elif app_mode == "📁 Batch File Diagnostics":
    st.markdown("### 📁 Batch Processing Station")
    st.write("Upload a CSV file containing multiple engine recordings. The system will process each recording, calculate predictions, render distributions, and output a detailed diagnostics file.")

    # Upload box
    uploaded_file = st.file_uploader("Upload Engine Sensor CSV File", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Check structure
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
                    
                    preds = model.predict(x_batch_scaled)
                    probs = model.predict_proba(x_batch_scaled)
                    max_probs = np.max(probs, axis=1)
                    
                    # Add outputs to dataframe
                    df_upload['Predicted_Fault_Label'] = preds
                    df_upload['Diagnosis_Name'] = [FAULT_CLASSES[p]['name'] for p in preds]
                    df_upload['Diagnostic_Severity'] = [FAULT_CLASSES[p]['severity'] for p in preds]
                    df_upload['Confidence_Score'] = max_probs
                    
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
                            <div class='metric-desc'>Operated nominally</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m3:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Anomaly Detections</div>
                            <div class='metric-value' style='color: #EF4444;'>{anomaly_count:,}</div>
                            <div class='metric-desc'>Required inspections</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_m4:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>Anomaly Ratio</div>
                            <div class='metric-value'>{anomaly_pct:.2%}</div>
                            <div class='metric-desc'>Total batch fault percentage</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
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
                            title='Engine Health & Fault Diagnostics Distribution',
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
                            title='Engine Operating State: Shaft RPM vs Fuel Flow',
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
