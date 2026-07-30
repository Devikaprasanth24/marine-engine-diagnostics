import pandas as pd
import numpy as np
import pickle
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")
dataset_path = "marine_engine_fault_dataset (1).csv"
data = pd.read_csv(dataset_path)

# Drop rows with missing values
data = data.dropna()
# Drop duplicate rows
data = data.drop_duplicates()

# Features (x) should drop Timestamp and Fault_Label
feature_cols = [col for col in data.columns if col not in ['Timestamp', 'Fault_Label']]
print("Feature columns used for training:")
print(feature_cols)

x = data[feature_cols].values
y = data['Fault_Label'].values

print("Splitting dataset...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Standardizing features...")
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# Save the scaler
print("Saving scaler to scaler.pkl...")
with open("scaler.pkl", "wb") as f:
    pickle.dump(sc, f)

# Dictionary of models to train
models_to_train = {
    'logistic_regression': {
        'name': 'Logistic Regression',
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'filename': 'logistic_model.pkl'
    },
    'random_forest': {
        'name': 'Random Forest',
        'model': RandomForestClassifier(n_estimators=100, random_state=42),
        'filename': 'random_forest_model.pkl'
    },
    'xgboost': {
        'name': 'XGBoost',
        'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'filename': 'xgboost_model.pkl'
    },
    'decision_tree': {
        'name': 'Decision Tree',
        'model': DecisionTreeClassifier(random_state=42),
        'filename': 'decision_tree_model.pkl'
    },
    'svm': {
        'name': 'Support Vector Machine',
        'model': SVC(kernel='rbf', probability=True, random_state=42),
        'filename': 'svm_model.pkl'
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'model': KNeighborsClassifier(n_neighbors=5),
        'filename': 'knn_model.pkl'
    }
}

metrics_dict = {}

# Train and evaluate each model
for model_key, info in models_to_train.items():
    print(f"\n--- Training {info['name']} ---")
    model = info['model']
    
    # Measure training time
    start_train = time.time()
    model.fit(x_train_scaled, y_train)
    end_train = time.time()
    train_time = end_train - start_train
    print(f"Training completed in {train_time:.4f} seconds")
    
    # Save the model
    print(f"Saving model to {info['filename']}...")
    with open(info['filename'], "wb") as f:
        pickle.dump(model, f)
        
    # Evaluate model
    print("Evaluating model...")
    # Measure inference time over test set
    start_inf = time.time()
    y_pred = model.predict(x_test_scaled)
    end_inf = time.time()
    inf_time = (end_inf - start_inf) / len(x_test_scaled) * 1000 # in ms per sample
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Average Inference Time: {inf_time:.6f} ms/sample")
    
    metrics_dict[model_key] = {
        'name': info['name'],
        'accuracy': float(acc),
        'train_time_seconds': float(train_time),
        'inference_time_ms_per_sample': float(inf_time),
        'cm': cm.tolist(),
        'report': report,
        'test_size': int(len(y_test))
    }

# Save all metrics to a single file for the Streamlit UI to load instantly
print("\nSaving metrics to model_metrics.pkl...")
with open("model_metrics.pkl", "wb") as f:
    pickle.dump({
        'metrics': metrics_dict,
        'feature_names': feature_cols
    }, f)

print("All models trained and saved successfully!")

