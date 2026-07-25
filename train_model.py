import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")
data = pd.read_csv("marine_engine_fault_dataset (1).csv")

# Drop rows with missing values
data = data.dropna()
# Drop duplicate rows
data = data.drop_duplicates()

# Features (x) should drop Timestamp and Fault_Label
# Column 0 is Timestamp, Column 11 is Fault_Label
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

print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train_scaled, y_train)

print("Evaluating model...")
y_pred = model.predict(x_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
print("Saving model and scaler to pickle files...")
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(sc, f)

print("Model and scaler saved successfully!")
