# Marine Engine Predictive Maintenance System ⚓

An interactive machine learning-powered web application designed to monitor, analyze, and diagnose faults in marine diesel engines. This tool utilizes multi-class classification models to identify various engine anomalies and provide actionable maintenance recommendations.

## 🚀 Live Demo
Once deployed, you can access the live application on Streamlit Community Cloud.

---

## 📋 Features

- **Multi-Model Diagnosis**: Evaluate and compare predictions from multiple trained models:
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Logistic Regression
- **Real-Time Anomaly Detection**: Input engine parameters (RPM, load, temperatures, vibrations) to immediately classify the engine status.
- **Interactive Visualizations**: Dynamic Plotly charts mapping:
  - Exhaust gas temperature profiles.
  - Axial and radial engine vibrations.
  - Fuel delivery vs. shaft speed.
  - Model metrics comparison (ROC curves, confusion matrices, precision/recall).
- **Severity & Recommendations**: Instantly generates warning levels (Healthy, Warning, Critical) along with precise troubleshooting recommendations.

---

## 🛠️ Fault Classes Diagnosed

1. **Normal Operation** (Healthy)
2. **Fuel Delivery System Anomaly** (Critical)
3. **Low Cylinder Compression Pressure** (Warning)
4. **Combustion Heat / Exhaust Gas Anomaly** (Warning)
5. **Radial Engine Vibration Fault** (Critical)
6. **Lubrication System Thermal Anomaly** (Critical)
7. **Air Intake Pressure / Turbocharger Fault** (Critical)
8. **Lubrication Pressure & Axial Vibration Fault** (Critical)

---

## 💻 Local Setup & Installation

Follow these steps to run the application locally on your machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Devikaprasanth24/marine-engine-diagnostics.git
   cd marine-engine-diagnostics
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Dataset
The models are trained on the `marine_engine_fault_dataset.csv`, which contains key thermodynamic and vibration features recorded from ship operations.
