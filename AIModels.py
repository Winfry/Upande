#WASH Site Anomaly Detection model using Isolation Forest with synthetic data generation and visualization:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ======================
# 1. Synthetic Data Generation
# ======================

def generate_wash_data(num_samples=1000):
    """Generate synthetic WASH site data with anomalies"""
    np.random.seed(42)
    
    # Generate normal data
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=num_samples, freq='H'),
        'water_level': np.random.normal(1.2, 0.2, num_samples),
        'ph': np.random.normal(7.0, 0.3, num_samples),
        'turbidity': np.abs(np.random.normal(2.0, 0.5, num_samples)),
        'temperature': np.random.normal(25.0, 1.5, num_samples),
        'ec': np.random.normal(800, 50, num_samples)
    }
    
    # Introduce 5% anomalies
    num_anomalies = int(num_samples * 0.05)
    anomaly_indices = np.random.choice(num_samples, num_anomalies, replace=False)
    
    # Create different types of anomalies
    data['water_level'][anomaly_indices] = np.random.uniform(0.1, 2.5, num_anomalies)
    data['ph'][anomaly_indices] = np.random.uniform(4.0, 10.0, num_anomalies)
    data['turbidity'][anomaly_indices] = np.random.uniform(10.0, 50.0, num_anomalies)
    data['temperature'][anomaly_indices] = np.random.uniform(-5.0, 40.0, num_anomalies)
    data['ec'][anomaly_indices] = np.random.uniform(100, 1500, num_anomalies)
    
    return pd.DataFrame(data)

# Generate synthetic dataset
df = generate_wash_data(1000)

# ======================
# 2. Data Preprocessing
# ======================

# Select features for modeling
features = ['water_level', 'ph', 'turbidity', 'temperature', 'ec']
X = df[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# 3. Model Training
# ======================

# Initialize Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_scaled)

# ======================
# 4. Anomaly Detection
# ======================

# Predict anomalies (-1 for anomalies, 1 for normal)
df['anomaly_score'] = model.decision_function(X_scaled)
df['is_anomaly'] = model.predict(X_scaled)
df['is_anomaly'] = np.where(df['is_anomaly'] == -1, 1, 0)  # Convert to 0/1

# ======================
# 5. Visualization
# ======================

def plot_anomalies(df, parameter):
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df[parameter], color='blue', label='Normal')
    plt.scatter(df[df['is_anomaly'] == 1]['timestamp'],
                df[df['is_anomaly'] == 1][parameter],
                color='red', label='Anomaly')
    plt.title(f'{parameter} Anomaly Detection')
    plt.xlabel('Timestamp')
    plt.ylabel(parameter)
    plt.legend()
    plt.show()

# Plot anomalies for each parameter
for param in features:
    plot_anomalies(df, param)

# ======================
# 6. Results Analysis
# ======================

print(f"Detected {df['is_anomaly'].sum()} anomalies")
print("\nSample Anomalies:")
print(df[df['is_anomaly'] == 1].head())

# ======================
# 7. Model Persistence
# ======================

import joblib

# Save model and scaler
joblib.dump(model, 'wash_anomaly_model.pkl')
joblib.dump(scaler, 'wash_scaler.pkl')

# Load model for later use
# model = joblib.load('wash_anomaly_model.pkl')
# scaler = joblib.load('wash_scaler.pkl')