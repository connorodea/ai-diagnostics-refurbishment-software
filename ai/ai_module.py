import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging

# Configure logging
logging.basicConfig(filename='../logs/ai_module.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    try:
        # Handle missing values
        data = data.dropna()

        # Feature Engineering: Example - Creating average CPU usage
        data['Average_CPU_Usage'] = data['cpu_usage'].apply(lambda x: np.mean(eval(x)) if isinstance(x, str) else 0)

        # Select features and target
        X = data[['Average_CPU_Usage', 'memory_usage', 'disk_health', 'gpu_temp', 'battery_health']]
        y = data['failure']  # 1 = Faulty, 0 = Healthy

        # Normalize numerical features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler for future use
        joblib.dump(scaler, '../ai/scaler.pkl')

        logging.info("Data preprocessed successfully.")
        return X_scaled, y
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None, None

def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Save the trained model
        joblib.dump(model, '../ai/failure_predictor.pkl')
        logging.info("Model trained and saved successfully.")
    except Exception as e:
        logging.error(f"Error training model: {e}")

def load_model():
    try:
        model = joblib.load('../ai/failure_predictor.pkl')
        scaler = joblib.load('../ai/scaler.pkl')
        logging.info("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        logging.error(f"Error loading model/scaler: {e}")
        return None, None

def predict_failure(cpu_usage, memory_usage, disk_health, gpu_temp, battery_health):
    try:
        model, scaler = load_model()
        if model and scaler:
            input_data = np.array([[cpu_usage, memory_usage, disk_health, gpu_temp, battery_health]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            result = {
                "Prediction": "Failure Detected" if prediction[0] == 1 else "Healthy",
                "Confidence": round(max(probability[0]) * 100, 2)
            }
            logging.info(f"Prediction made: {result}")
            return result
        else:
            logging.error("Model or scaler not loaded.")
            return {"Prediction": "Error", "Confidence": 0}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return {"Prediction": "Error", "Confidence": 0}
