import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error

class TemperaturePredictionModel:
    def __init__(self, window_size=7, forecast_horizon=7):
        """
        Initialize the Temperature Prediction Model
        
        :param window_size: Number of previous days to use for prediction
        :param forecast_horizon: Number of days to forecast
        """
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler()
        self.model = None

    def generate_synthetic_data(self, n_days=100):
        """
        Generate synthetic temperature data with seasonal and random variations
        
        :param n_days: Number of days to generate data for
        :return: Pandas DataFrame with temperature data
        """
        np.random.seed(42)
        
        # Simulate seasonal temperature variation
        days = np.arange(n_days)
        base_temp = 20  # Base temperature
        seasonal_variation = 10 * np.sin(2 * np.pi * days / 365)  # Annual cycle
        random_noise = np.random.normal(0, 2, n_days)  # Random fluctuations
        
        temperatures = base_temp + seasonal_variation + random_noise
        
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=n_days),
            'temperature': temperatures
        })
        
        return df

    def prepare_data(self, data):
        """
        Prepare data for LSTM model
        
        :param data: Input temperature data
        :return: Prepared input and output sequences
        """
        # Normalize the data
        scaled_data = self.scaler.fit_transform(data['temperature'].values.reshape(-1, 1))
        
        # Create input sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.window_size - self.forecast_horizon + 1):
            X.append(scaled_data[i:i+self.window_size])
            y.append(scaled_data[i+self.window_size:i+self.window_size+self.forecast_horizon])
        
        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape):
        """
        Build LSTM model for temperature prediction
        
        :param input_shape: Shape of input data
        :return: Compiled LSTM model
        """
        model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the LSTM model
        
        :param X_train: Training input sequences
        :param y_train: Training target sequences
        :param epochs: Number of training epochs
        :param batch_size: Batch size for training
        """
        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, X_test):
        """
        Make predictions using the trained model
        
        :param X_test: Input test sequences
        :return: Predicted temperatures
        """
        predictions = self.model.predict(X_test)
        return self.scaler.inverse_transform(predictions.reshape(-1, 1))

    def evaluate_model(self, y_test, predictions):
        """
        Evaluate model performance
        
        :param y_test: Actual test values
        :param predictions: Predicted values
        :return: Performance metrics
        """
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        return {
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae
        }

    def visualize_predictions(self, actual, predicted, title='Temperature Prediction'):
        """
        Visualize actual vs predicted temperatures
        
        :param actual: Actual temperature values
        :param predicted: Predicted temperature values
        :param title: Plot title
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual Temperature', color='blue')
        plt.plot(predicted, label='Predicted Temperature', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Temperature')
        plt.legend()
        plt.show()

def main():
    # Initialize and run the temperature prediction model
    temp_model = TemperaturePredictionModel(window_size=7, forecast_horizon=7)
    
    # Generate synthetic data
    data = temp_model.generate_synthetic_data(n_days=200)
    
    # Prepare data for LSTM
    X, y = temp_model.prepare_data(data)
    
    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Train the model
    temp_model.train_model(X_train, y_train)
    
    # Make predictions
    predictions = temp_model.predict(X_test)
    
    # Evaluate model
    metrics = temp_model.evaluate_model(
        temp_model.scaler.inverse_transform(y_test.reshape(-1, 1)), 
        predictions
    )
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    
    # Visualize predictions
    temp_model.visualize_predictions(
        temp_model.scaler.inverse_transform(y_test.reshape(-1, 1)), 
        predictions
    )

if __name__ == "__main__":
    main()