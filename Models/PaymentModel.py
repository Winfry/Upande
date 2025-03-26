import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    roc_auc_score
)

class PaymentMeterControlModel:
    def __init__(self):
        """
        Initialize the Payment-Based Meter Control Model
        """
        self.model = None
        self.scaler = StandardScaler()

    def generate_synthetic_data(self, n_samples=100):
        """
        Generate synthetic training data for payment-based meter control
        
        :param n_samples: Number of data samples to generate
        :return: Pandas DataFrame with payment and reporting frequency data
        """
        np.random.seed(42)
        
        # Generate random payment amounts
        payment_amounts = np.random.uniform(1000, 10000, n_samples)
        
        # Create decision logic for reporting frequency increase
        # More complex decision logic can be implemented here
        reporting_increase = (
            (payment_amounts > 5000) & (np.random.random(n_samples) > 0.5)
        ).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'payment_amount': payment_amounts,
            'reporting_increase': reporting_increase
        })
        
        return df

    def prepare_data(self, data):
        """
        Prepare data for logistic regression model
        
        :param data: Input DataFrame
        :return: Prepared features and target variables
        """
        X = data['payment_amount'].values.reshape(-1, 1)
        y = data['reporting_increase'].values
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

    def train_model(self, X_train, y_train):
        """
        Train logistic regression model
        
        :param X_train: Training features
        :param y_train: Training labels
        """
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions using the trained model
        
        :param X_test: Test features
        :return: Predictions and probabilities
        """
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]
        
        return predictions, probabilities

    def evaluate_model(self, y_test, predictions):
        """
        Evaluate model performance
        
        :param y_test: Actual test labels
        :param predictions: Predicted labels
        :return: Performance metrics
        """
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions)
        
        return {
            'Accuracy': accuracy,
            'Confusion Matrix': conf_matrix,
            'Classification Report': class_report
        }

    def plot_roc_curve(self, y_test, probabilities):
        """
        Plot Receiver Operating Characteristic (ROC) curve
        
        :param y_test: Actual test labels
        :param probabilities: Predicted probabilities
        """
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        auc_score = roc_auc_score(y_test, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

    def interpret_model(self):
        """
        Interpret logistic regression model coefficients
        
        :return: Model interpretation dictionary
        """
        return {
            'Intercept': self.model.intercept_[0],
            'Coefficient': self.model.coef_[0][0],
            'Interpretation': (
                "A positive coefficient means higher payment amounts "
                "increase the likelihood of reporting frequency change."
            )
        }

def main():
    # Initialize the Payment Meter Control Model
    meter_model = PaymentMeterControlModel()
    
    # Generate synthetic data
    data = meter_model.generate_synthetic_data(n_samples=200)
    
    # Prepare data
    X, y = meter_model.prepare_data(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train the model
    meter_model.train_model(X_train, y_train)
    
    # Make predictions
    predictions, probabilities = meter_model.predict(X_test)
    
    # Evaluate model performance
    metrics = meter_model.evaluate_model(y_test, predictions)
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}:\n{value}\n")
    
    # Plot ROC Curve
    meter_model.plot_roc_curve(y_test, probabilities)
    
    # Model Interpretation
    interpretation = meter_model.interpret_model()
    print("Model Interpretation:")
    for key, value in interpretation.items():
        print(f"{key}: {value}")
    
    # Example Prediction
    example_payment = np.array([[6000]])  # Example payment amount
    example_payment_scaled = meter_model.scaler.transform(example_payment)
    example_prediction, example_prob = meter_model.predict(example_payment_scaled)
    
    print("\nExample Prediction:")
    print(f"Payment Amount: KES {example_payment[0][0]}")
    print(f"Reporting Frequency Increase: {bool(example_prediction[0])}")
    print(f"Probability of Increase: {example_prob[0]:.2%}")

if __name__ == "__main__":
    main()