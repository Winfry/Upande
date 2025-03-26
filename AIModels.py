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


# Temperature Prediction
class TemperaturePrediction:
    def __init__(self):
        """
        2. Temperature Prediction
        2.1 Purpose
        The Temperature Prediction model provides insights into temperature variations over time, which is crucial for monitoring environmental conditions at WASH sites.
        
        2.2 Model Used
        A placeholder model was implemented using random temperature variations. This can be replaced with a time-series forecasting model such as ARIMA or LSTMs for better accuracy.
        
        2.3 Data
        A dataset with 100 days of temperature values was generated. This data helps in detecting trends and understanding seasonal variations affecting water quality.
        """
        self.data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='D'),
            'temperature': np.random.normal(25, 5, 100)
        })
        
        # Placeholder ARIMA Model
        self.model = ARIMA(self.data['temperature'], order=(1,1,1))
        self.model_fitted = self.model.fit()
    
    def predict_temperature(self, steps=24):
        """Generates future temperature predictions for the given steps."""
        forecast = self.model_fitted.forecast(steps=steps)
        return forecast



# Payment-Based Meter Control
class PaymentMeterControl:
    def __init__(self):
        """
        Optimizes smart meter reporting frequency based on user payment behavior.
        Uses a Logistic Regression model to classify whether to increase reporting frequency.
        """
        self.data = pd.DataFrame({
            'customer_id': np.arange(1, 101),
            'amount_paid': np.random.uniform(100, 10000, 100),
            'increase_frequency': np.random.choice([0, 1], size=100)
        })
        
        # Feature Scaling and Model Training
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        X = self.data[['amount_paid']]
        y = self.data['increase_frequency']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
    
    def predict(self, amount):
        """Predicts whether meter reporting frequency should increase based on payment."""
        scaled_amount = self.scaler.transform(np.array([[amount]]))
        return self.model.predict(scaled_amount)[0]

# Example Usage
if __name__ == "__main__":
    wash_model = WashAnomalyDetection()
    temp_model = TemperaturePrediction()
    payment_model = PaymentMeterControl()
    
    # Example payment prediction
    example_payment = 5000
    prediction = payment_model.predict(example_payment)
    print(f'Predicted meter frequency change: {prediction}')
    
    # Example temperature forecast
    temp_forecast = temp_model.predict_temperature(steps=10)
    print(f'Temperature forecast: {temp_forecast}')
