#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to create directories
create_directories() {
    echo "Creating directory structure..."
    mkdir -p diagnostics
    mkdir -p ai
    mkdir -p gui
    mkdir -p web_interface
    mkdir -p api
    mkdir -p integration
    mkdir -p security
    mkdir -p inventory
    mkdir -p tests
    mkdir -p config
    mkdir -p logs
    echo "Directories created successfully."
}

# Function to create Python files with content
create_python_files() {
    echo "Creating Python source files..."

    # diagnostics/diagnostics.py
    cat <<EOF > diagnostics/diagnostics.py
import psutil
import platform
import wmi
import random  # Placeholder for actual hardware data
from pySMART import Smart
import GPUtil
import serial
import logging

# Configure logging
logging.basicConfig(filename='../logs/diagnostics.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class CPUDiagnostics:
    def __init__(self):
        self.system = platform.system()

    def get_cpu_info(self):
        try:
            cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
            cpu_freq = psutil.cpu_freq()._asdict()
            cpu_temp = self.get_cpu_temperature()
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else "N/A"
            return {
                "CPU Usage (%)": cpu_usage,
                "CPU Frequency (MHz)": cpu_freq,
                "CPU Temperature (°C)": cpu_temp,
                "Load Average": load_avg
            }
        except Exception as e:
            logging.error(f"CPU Diagnostics Error: {e}")
            return {"Error": str(e)}

    def get_cpu_temperature(self):
        try:
            if self.system == "Windows":
                w = wmi.WMI(namespace="root\\wmi")
                temperature_info = w.MSAcpi_ThermalZoneTemperature()[0]
                return (temperature_info.CurrentTemperature / 10.0) - 273.15
            elif self.system == "Linux":
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp = float(f.read()) / 1000.0
                    return temp
            elif self.system == "Darwin":
                # macOS-specific temperature retrieval can be implemented using third-party tools or APIs
                return "N/A"
            else:
                return "N/A"
        except Exception as e:
            logging.error(f"CPU Temperature Retrieval Error: {e}")
            return f"Error: {e}"

class RAMDiagnostics:
    def get_memory_info(self):
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                "Total Memory (GB)": round(memory.total / (1024 ** 3), 2),
                "Available Memory (GB)": round(memory.available / (1024 ** 3), 2),
                "Memory Usage (%)": memory.percent,
                "Total Swap (GB)": round(swap.total / (1024 ** 3), 2),
                "Swap Usage (%)": swap.percent
            }
        except Exception as e:
            logging.error(f"RAM Diagnostics Error: {e}")
            return {"Error": str(e)}

class StorageDiagnostics:
    def get_storage_info(self):
        try:
            partitions = psutil.disk_partitions()
            storage_info = {}
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)._asdict()
                    io = psutil.disk_io_counters(perdisk=True)
                    disk_name = partition.device.split('/')[-1]
                    storage_info[partition.device] = {
                        "Mountpoint": partition.mountpoint,
                        "File System": partition.fstype,
                        "Usage": usage,
                        "IO Statistics": io.get(disk_name, "N/A")
                    }
                except PermissionError:
                    continue
            smart_info = self.get_smart_info()
            storage_info["SMART"] = smart_info
            return storage_info
        except Exception as e:
            logging.error(f"Storage Diagnostics Error: {e}")
            return {"Error": str(e)}

    def get_smart_info(self):
        try:
            drives = Smart.devices()
            smart_data = {}
            for drive in drives:
                smart_data[drive.name] = {
                    "Model": drive.model,
                    "Serial": drive.serial,
                    "Health": drive.assessment,
                    "Temperature": getattr(drive, 'temperature', 'N/A'),
                    "Reallocated Sectors": getattr(drive, 'reallocated_sector_count', 'N/A')
                }
            return smart_data
        except Exception as e:
            logging.error(f"SMART Data Retrieval Error: {e}")
            return f"Error retrieving SMART data: {e}"

class GPUDiagnostics:
    def get_gpu_info(self):
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = {}
            for gpu in gpus:
                gpu_info[gpu.id] = {
                    "Name": gpu.name,
                    "Load (%)": round(gpu.load * 100, 2),
                    "Free Memory (MB)": gpu.memoryFree,
                    "Used Memory (MB)": gpu.memoryUsed,
                    "Total Memory (MB)": gpu.memoryTotal,
                    "Temperature (°C)": gpu.temperature,
                    "UUID": gpu.uuid,
                    "Clock Speed (MHz)": gpu.clock,
                    "Fan Speed (%)": gpu.fan,
                }
            return gpu_info
        except Exception as e:
            logging.error(f"GPU Diagnostics Error: {e}")
            return {"Error": str(e)}

class BatteryDiagnostics:
    def get_battery_info(self):
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    "Battery Percent": battery.percent,
                    "Power Plugged In": battery.power_plugged,
                    "Battery Time Remaining (seconds)": battery.secsleft,
                    "Battery Health (%)": self.get_battery_health(),
                    "Charge Cycles": self.get_charge_cycles()
                }
            else:
                return {"Battery": "Not Available"}
        except Exception as e:
            logging.error(f"Battery Diagnostics Error: {e}")
            return {"Error": str(e)}

    def get_battery_health(self):
        # Implement actual battery health retrieval based on hardware APIs or external tools
        # Placeholder implementation
        return random.randint(70, 100)

    def get_charge_cycles(self):
        # Implement actual charge cycle retrieval based on hardware APIs or external tools
        # Placeholder implementation
        return random.randint(100, 500)

class PowerSupplyDiagnostics:
    def get_power_supply_info(self):
        try:
            # Placeholder for actual implementation
            # Requires interfacing with hardware via serial or other protocols
            voltage, stability = self.read_power_supply_voltage()
            return {
                "Power Supply Voltage (V)": voltage,
                "Voltage Stability": stability
            }
        except Exception as e:
            logging.error(f"Power Supply Diagnostics Error: {e}")
            return {"Error": str(e)}

    def read_power_supply_voltage(self):
        # Simulated voltage reading; replace with actual hardware communication
        # Example using pyserial to communicate with a multimeter or power supply unit
        try:
            ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Update COM port as needed
            ser.write(b'READ_VOLTS\n')  # Command depends on your device's protocol
            response = ser.readline().decode('utf-8').strip()
            ser.close()
            voltage = float(response)
            stability = self.assess_voltage_stability(voltage)
            return voltage, stability
        except Exception as e:
            logging.error(f"Power Supply Reading Error: {e}")
            return "N/A", "N/A"

    def assess_voltage_stability(self, voltage):
        # Placeholder for actual voltage stability assessment
        if 11.5 <= voltage <= 12.5:
            return "Stable"
        elif 10.0 <= voltage < 11.5 or 12.5 < voltage <= 13.5:
            return "Marginal"
        else:
            return "Unstable"
EOF

    # ai/ai_module.py
    cat <<EOF > ai/ai_module.py
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
EOF

    # ai/model_optimization.py
    cat <<EOF > ai/model_optimization.py
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

# Configure logging
logging.basicConfig(filename='../logs/model_optimization.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def optimize_model(X_train, y_train):
    try:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Best Score: {grid_search.best_score_}")
        return grid_search.best_estimator_
    except Exception as e:
        logging.error(f"Model Optimization Error: {e}")
        return None

if __name__ == "__main__":
    from ai_module import load_data, preprocess_data, train_model
    data = load_data('../data/diagnostic_data.csv')
    if data is not None:
        X, y = preprocess_data(data)
        if X is not None and y is not None:
            optimized_model = optimize_model(X, y)
            if optimized_model:
                joblib.dump(optimized_model, '../ai/optimized_failure_predictor.pkl')
                logging.info("Optimized model saved successfully.")
EOF

    # ai/model_explainability.py
    cat <<EOF > ai/model_explainability.py
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(filename='../logs/model_explainability.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def explain_model():
    try:
        model = joblib.load('../ai/optimized_failure_predictor.pkl')
        scaler = joblib.load('../ai/scaler.pkl')
        data = pd.read_csv('../data/diagnostic_data.csv').dropna()
        X = data[['Average_CPU_Usage', 'memory_usage', 'disk_health', 'gpu_temp', 'battery_health']]
        X_scaled = scaler.transform(X)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        # Summary Plot
        shap.summary_plot(shap_values, X, plot_type="bar")
        plt.savefig('../reports/shap_summary_plot.png')
        plt.clf()
        logging.info("SHAP summary plot generated and saved.")

        # Force Plot for the first prediction
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0], matplotlib=True)
        plt.savefig('../reports/shap_force_plot.png')
        plt.clf()
        logging.info("SHAP force plot generated and saved.")
    except Exception as e:
        logging.error(f"Model Explainability Error: {e}")

if __name__ == "__main__":
    explain_model()
EOF

    # gui/gui.py
    cat <<EOF > gui/gui.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QMessageBox
)
import json
from workflow import execute_workflow

class DiagnosticApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Electronics Diagnostic Tool')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.status_label = QLabel('Status: Ready')
        layout.addWidget(self.status_label)

        self.run_button = QPushButton('Run Diagnostics')
        self.run_button.clicked.connect(self.run_diagnostics)
        layout.addWidget(self.run_button)

        self.report_area = QTextEdit()
        self.report_area.setReadOnly(True)
        layout.addWidget(self.report_area)

        self.setLayout(layout)

    def run_diagnostics(self):
        reply = QMessageBox.question(
            self, 'Confirm', 'Are you sure you want to run diagnostics?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.status_label.setText('Status: Running Diagnostics...')
            QApplication.processEvents()  # Update UI
            report = execute_workflow()
            if "Error" not in report:
                self.report_area.setText(json.dumps(report, indent=4))
                self.status_label.setText('Status: Diagnostics Completed')
                QMessageBox.information(self, 'Success', 'Diagnostics completed successfully.')
            else:
                self.report_area.setText(json.dumps(report, indent=4))
                self.status_label.setText('Status: Diagnostics Failed')
                QMessageBox.critical(self, 'Error', 'Diagnostics failed. Check logs for details.')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DiagnosticApp()
    ex.show()
    sys.exit(app.exec_())
EOF

    # web_interface/web_interface.py
    cat <<EOF > web_interface/web_interface.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import json
from workflow import execute_workflow

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Electronics Diagnostic Dashboard"), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Run Diagnostics", id="run-button", color="primary"), width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Loading(id="loading", children=[html.Div(id="report-output")], type="default"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='cpu-usage'), width=6),
        dbc.Col(dcc.Graph(id='memory-usage'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='disk-usage'), width=6),
        dbc.Col(dcc.Graph(id='gpu-temp'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='battery-health'), width=6),
        dbc.Col(dcc.Graph(id='power-supply'), width=6)
    ])
], fluid=True)

@app.callback(
    Output("report-output", "children"),
    [
        Input("run-button", "n_clicks")
    ]
)
def run_diagnostics(n_clicks):
    if n_clicks:
        report = execute_workflow()
        if "Error" not in report:
            return html.Pre(json.dumps(report, indent=4))
        else:
            return html.Pre(json.dumps(report, indent=4))
    return ""

@app.callback(
    [
        Output('cpu-usage', 'figure'),
        Output('memory-usage', 'figure'),
        Output('disk-usage', 'figure'),
        Output('gpu-temp', 'figure'),
        Output('battery-health', 'figure'),
        Output('power-supply', 'figure')
    ],
    [Input("run-button", "n_clicks")]
)
def update_graphs(n_clicks):
    if n_clicks:
        report = execute_workflow()
        if "Error" not in report:
            cpu = report['Diagnostics']['CPU']['CPU Usage (%)']
            memory = report['Diagnostics']['Memory']['Memory Usage (%)']
            # Example: Extract disk read count
            disk_io = report['Diagnostics']['Storage']['IO Statistics']
            disk_read = disk_io['sda']['read_count'] if 'sda' in disk_io else 0
            gpu_temp = report['Diagnostics']['GPU']['0']['Temperature (°C)'] if '0' in report['Diagnostics']['GPU'] else 0
            battery = report['Diagnostics']['Battery']['Battery Health (%)'] if 'Battery Health (%)' in report['Diagnostics']['Battery'] else 100
            power = report['Diagnostics']['Power Supply']['Power Supply Voltage (V)'] if 'Power Supply Voltage (V)' in report['Diagnostics']['Power Supply'] else 0

            # Create figures
            cpu_fig = {
                'data': [{'x': list(range(len(cpu))), 'y': cpu, 'type': 'line', 'name': 'CPU Usage'}],
                'layout': {'title': 'CPU Usage (%)'}
            }
            memory_fig = {
                'data': [{'x': list(range(len(memory))), 'y': memory, 'type': 'line', 'name': 'Memory Usage'}],
                'layout': {'title': 'Memory Usage (%)'}
            }
            disk_fig = {
                'data': [{'x': ['Read Count'], 'y': [disk_read], 'type': 'bar', 'name': 'Disk Read Count'}],
                'layout': {'title': 'Disk Read Count'}
            }
            gpu_fig = {
                'data': [{'x': ['GPU Temperature'], 'y': [gpu_temp], 'type': 'bar', 'name': 'GPU Temp'}],
                'layout': {'title': 'GPU Temperature (°C)'}
            }
            battery_fig = {
                'data': [{'labels': ['Healthy', 'Needs Replacement'], 'values': [battery, 100 - battery], 'type': 'pie'}],
                'layout': {'title': 'Battery Health (%)'}
            }
            power_fig = {
                'data': [{'x': ['Voltage'], 'y': [power], 'type': 'bar', 'name': 'Power Supply Voltage'}],
                'layout': {'title': 'Power Supply Voltage (V)'}
            }

            return cpu_fig, memory_fig, disk_fig, gpu_fig, battery_fig, power_fig
    return {}, {}, {}, {}, {}, {}

if __name__ == '__main__':
    app.run_server(debug=True)
EOF

    # api/api.py
    cat <<EOF > api/api.py
from flask import Flask, jsonify, request
from workflow import execute_workflow
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='../logs/api.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

@app.route('/run_diagnostics', methods=['POST'])
def run_diagnostics_api():
    try:
        report = execute_workflow()
        return jsonify(report), 200
    except Exception as e:
        logging.error(f"API Diagnostics Error: {e}")
        return jsonify({"Error": str(e)}), 500

@app.route('/get_report/<report_id>', methods=['GET'])
def get_report(report_id):
    # Implement logic to retrieve report by ID from the database or storage
    # Placeholder response
    return jsonify({"report_id": report_id, "data": "Report data here"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
EOF

    # integration/hardware_integration.py
    cat <<EOF > integration/hardware_integration.py
import serial
import logging

# Configure logging
logging.basicConfig(filename='../logs/hardware_integration.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class Multimeter:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

    def read_voltage(self):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=self.timeout) as ser:
                ser.write(b'READ_VOLTS\n')  # Command depends on your device's protocol
                response = ser.readline().decode('utf-8').strip()
                voltage = float(response)
                logging.info(f"Voltage Read: {voltage} V")
                return voltage
        except Exception as e:
            logging.error(f"Multimeter Read Voltage Error: {e}")
            return "N/A"

    def read_current(self):
        try:
            with serial.Serial(self.port, self.baudrate, timeout=self.timeout) as ser:
                ser.write(b'READ_CURRENT\n')  # Command depends on your device's protocol
                response = ser.readline().decode('utf-8').strip()
                current = float(response)
                logging.info(f"Current Read: {current} A")
                return current
        except Exception as e:
            logging.error(f"Multimeter Read Current Error: {e}")
            return "N/A"
EOF

    # integration/cloud_storage.py
    cat <<EOF > integration/cloud_storage.py
import boto3
from botocore.exceptions import NoCredentialsError
import logging

# Configure logging
logging.basicConfig(filename='../logs/cloud_storage.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def upload_to_s3(file_name, bucket, object_name=None):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, object_name or file_name)
        logging.info(f"File {file_name} uploaded to {bucket} as {object_name or file_name}")
    except FileNotFoundError:
        logging.error(f"The file {file_name} was not found.")
    except NoCredentialsError:
        logging.error("Credentials not available.")
EOF

    # security/security.py
    cat <<EOF > security/security.py
from cryptography.fernet import Fernet
import os
import logging

# Configure logging
logging.basicConfig(filename='../logs/security.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def generate_key():
    key = Fernet.generate_key()
    with open("../config/secret.key", "wb") as key_file:
        key_file.write(key)
    logging.info("Encryption key generated and saved.")

def load_key():
    try:
        return open("../config/secret.key", "rb").read()
    except Exception as e:
        logging.error(f"Load Key Error: {e}")
        return None

def encrypt_data(data, key):
    try:
        f = Fernet(key)
        encrypted = f.encrypt(data.encode())
        logging.info("Data encrypted successfully.")
        return encrypted
    except Exception as e:
        logging.error(f"Encrypt Data Error: {e}")
        return None

def decrypt_data(encrypted_data, key):
    try:
        f = Fernet(key)
        decrypted = f.decrypt(encrypted_data).decode()
        logging.info("Data decrypted successfully.")
        return decrypted
    except Exception as e:
        logging.error(f"Decrypt Data Error: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists("../config/secret.key"):
        generate_key()
    key = load_key()
    sample_data = "Sensitive Diagnostic Report Data"
    encrypted = encrypt_data(sample_data, key)
    print(f"Encrypted: {encrypted}")
    decrypted = decrypt_data(encrypted, key)
    print(f"Decrypted: {decrypted}")
EOF

    # inventory/inventory_integration.py
    cat <<EOF > inventory/inventory_integration.py
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# Configure logging
logging.basicConfig(filename='../logs/inventory_integration.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

Base = declarative_base()

class Part(Base):
    __tablename__ = 'parts'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    quantity = Column(Integer, nullable=False)
    cost = Column(Float, nullable=False)

def update_inventory(part_name, quantity_used):
    try:
        engine = create_engine('postgresql://user:password@localhost/dbname')  # Update with your DB credentials
        Session = sessionmaker(bind=engine)
        session = Session()
        part = session.query(Part).filter_by(name=part_name).first()
        if part and part.quantity >= quantity_used:
            part.quantity -= quantity_used
            session.commit()
            logging.info(f"Updated inventory for {part_name}: -{quantity_used}")
        else:
            logging.warning(f"Insufficient inventory for {part_name}")
        session.close()
    except Exception as e:
        logging.error(f"Inventory Update Error: {e}")

if __name__ == "__main__":
    update_inventory('RAM Module', 2)
EOF

    # tests/test_diagnostics.py
    cat <<EOF > tests/test_diagnostics.py
import unittest
from diagnostics.diagnostics import CPUDiagnostics, RAMDiagnostics, StorageDiagnostics, GPUDiagnostics, BatteryDiagnostics, PowerSupplyDiagnostics

class TestDiagnostics(unittest.TestCase):

    def setUp(self):
        self.cpu = CPUDiagnostics()
        self.ram = RAMDiagnostics()
        self.storage = StorageDiagnostics()
        self.gpu = GPUDiagnostics()
        self.battery = BatteryDiagnostics()
        self.power = PowerSupplyDiagnostics()

    def test_cpu_info(self):
        cpu_info = self.cpu.get_cpu_info()
        self.assertIn("CPU Usage (%)", cpu_info)
        self.assertIn("CPU Frequency (MHz)", cpu_info)
        self.assertIn("CPU Temperature (°C)", cpu_info)

    def test_memory_info(self):
        memory_info = self.ram.get_memory_info()
        self.assertIn("Total Memory (GB)", memory_info)
        self.assertIn("Available Memory (GB)", memory_info)
        self.assertIn("Memory Usage (%)", memory_info)

    def test_storage_info(self):
        storage_info = self.storage.get_storage_info()
        self.assertIn("SMART", storage_info)

    def test_gpu_info(self):
        gpu_info = self.gpu.get_gpu_info()
        # Depending on system, GPU info may or may not be available
        # So, check if it's either a dictionary or an error message
        self.assertTrue(isinstance(gpu_info, dict) or "Error" in gpu_info)

    def test_battery_info(self):
        battery_info = self.battery.get_battery_info()
        # Battery info might not be available on desktops
        self.assertTrue(isinstance(battery_info, dict))

    def test_power_supply_info(self):
        power_info = self.power.get_power_supply_info()
        self.assertIn("Power Supply Voltage (V)", power_info)

if __name__ == '__main__':
    unittest.main()
EOF

    # tests/test_integration.py
    cat <<EOF > tests/test_integration.py
import unittest
from workflow import run_all_diagnostics, generate_report, save_report, log_diagnostics, notify_technician

class TestIntegration(unittest.TestCase):

    def test_full_diagnostics_workflow(self):
        report = run_all_diagnostics()
        self.assertIn("CPU", report["Diagnostics"])
        self.assertIn("Memory", report["Diagnostics"])
        self.assertIn("Storage", report["Diagnostics"])
        self.assertIn("GPU", report["Diagnostics"])
        self.assertIn("Battery", report["Diagnostics"])
        self.assertIn("Power Supply", report["Diagnostics"])
        self.assertIn("AI Prediction", report)

if __name__ == '__main__':
    unittest.main()
EOF

    # tests/test_workflow.py
    cat <<EOF > tests/test_workflow.py
import unittest
from workflow import execute_workflow

class TestWorkflow(unittest.TestCase):

    def test_execute_workflow(self):
        report = execute_workflow()
        self.assertIn("Timestamp", report)
        self.assertIn("Diagnostics", report)
        self.assertIn("AI Prediction", report)
        self.assertIn("CPU", report["Diagnostics"])
        self.assertIn("Memory", report["Diagnostics"])
        self.assertIn("Storage", report["Diagnostics"])

if __name__ == '__main__':
    unittest.main()
EOF

    # workflow.py
    cat <<EOF > workflow.py
from diagnostics.diagnostics import CPUDiagnostics, RAMDiagnostics, StorageDiagnostics, GPUDiagnostics, BatteryDiagnostics, PowerSupplyDiagnostics
import json
from datetime import datetime
import logging
from ai.ai_module import predict_failure
from integration.cloud_storage import upload_to_s3
from security.security import encrypt_data, load_key
import smtplib
from email.mime.text import MIMEText

# Configure logging
logging.basicConfig(filename='logs/workflow.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def run_all_diagnostics():
    diagnostics = {}
    cpu = CPUDiagnostics()
    ram = RAMDiagnostics()
    storage = StorageDiagnostics()
    gpu = GPUDiagnostics()
    battery = BatteryDiagnostics()
    power = PowerSupplyDiagnostics()

    diagnostics["CPU"] = cpu.get_cpu_info()
    diagnostics["Memory"] = ram.get_memory_info()
    diagnostics["Storage"] = storage.get_storage_info()
    diagnostics["GPU"] = gpu.get_gpu_info()
    diagnostics["Battery"] = battery.get_battery_info()
    diagnostics["Power Supply"] = power.get_power_supply_info()

    return diagnostics

def generate_report(diagnostics, prediction):
    report = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Diagnostics": diagnostics,
        "AI Prediction": prediction
    }
    return report

def save_report(report, filename="diagnostic_report.json"):
    try:
        with open(filename, "w") as file:
            json.dump(report, file, indent=4)
        logging.info(f"Report saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving report: {e}")

def log_diagnostics(report):
    try:
        with open('logs/diagnostic_logs.log', 'a') as f:
            f.write(json.dumps(report) + "\n")
        logging.info("Diagnostics logged successfully.")
    except Exception as e:
        logging.error(f"Error logging diagnostics: {e}")

def notify_technician(report, technician_email="technician@example.com"):
    try:
        subject = f"Diagnostic Report - {report['Timestamp']}"
        body = json.dumps(report, indent=4)

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = "your_email@example.com"  # Replace with your email
        msg['To'] = technician_email

        with smtplib.SMTP_SSL('smtp.example.com', 465) as server:  # Replace with your SMTP server
            server.login("your_email@example.com", "your_password")  # Replace with your credentials
            server.sendmail("your_email@example.com", technician_email, msg.as_string())
        logging.info(f"Notification sent to {technician_email}")
    except Exception as e:
        logging.error(f"Error sending notification: {e}")

def execute_workflow():
    try:
        # Step 1: Run Diagnostics
        diagnostics = run_all_diagnostics()
        logging.info("Diagnostics run successfully.")

        # Step 2: AI Prediction
        prediction_input = [
            diagnostics['CPU']['CPU Usage (%)'],
            diagnostics['Memory']['Memory Usage (%)'],
            diagnostics['Storage']['SMART']['sda']['Reallocated Sectors'] if 'sda' in diagnostics['Storage']['SMART'] else 0,
            diagnostics['GPU']['0']['Temperature (°C)'] if '0' in diagnostics['GPU'] else 0,
            diagnostics['Battery']['Battery Health (%)'] if 'Battery Health (%)' in diagnostics['Battery'] else 100
        ]
        prediction = predict_failure(*prediction_input)
        logging.info(f"AI Prediction: {prediction}")

        # Step 3: Generate Report
        report = generate_report(diagnostics, prediction)
        logging.info("Report generated successfully.")

        # Step 4: Save and Log Report
        save_report(report)
        log_diagnostics(report)
        logging.info("Report saved and logged successfully.")

        # Step 5: Upload to Cloud Storage
        upload_to_s3('diagnostic_report.json', 'your-s3-bucket-name')  # Replace with your S3 bucket
        logging.info("Report uploaded to cloud storage successfully.")

        # Step 6: Encrypt Report
        key = load_key()
        encrypted_report = encrypt_data(json.dumps(report), key)
        with open('diagnostic_report_encrypted.json', 'wb') as f:
            f.write(encrypted_report)
        logging.info("Report encrypted successfully.")

        # Step 7: Notify Technician
        notify_technician(report)
        logging.info("Technician notified successfully.")

        return report
    except Exception as e:
        logging.error(f"Workflow Execution Error: {e}")
        return {"Error": str(e)}

if __name__ == "__main__":
    report = execute_workflow()
    print("Workflow Execution Completed:")
    print(report)
EOF

    # Dockerfile
    cat <<EOF > Dockerfile
# Use official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port for Dash app
EXPOSE 8050

# Command to run the web interface
CMD ["python", "web_interface/web_interface.py"]
EOF

    # requirements.txt
    cat <<EOF > requirements.txt
psutil
pySMART
py-cpuinfo
wmi; platform_system == "Windows"
GPUtil
pyserial
dash
dash-bootstrap-components
Flask
SQLAlchemy
Flask-Bcrypt
Flask-JWT-Extended
scikit-learn
pandas
numpy
joblib
shap
PyQt5
boto3
cryptography
EOF

    # README.md
    cat <<EOF > README.md
# Electronics Diagnostic Tool

## Overview

A comprehensive diagnostic software for electronics testing and refurbishment, integrating advanced AI for predictive analytics.

## Directory Structure

- `diagnostics/` - Hardware diagnostic modules
- `ai/` - AI and machine learning modules
- `gui/` - Desktop GUI application
- `web_interface/` - Web-based dashboard
- `api/` - RESTful API services
- `integration/` - External tool integrations
- `security/` - Data encryption and security modules
- `inventory/` - Inventory management integrations
- `tests/` - Unit and integration tests
- `config/` - Configuration files
- `logs/` - Log files
- `reports/` - Generated reports
- `Dockerfile` - Docker configuration for containerization
- `requirements.txt` - Python dependencies

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ElectronicsDiagnosticTool.git
   cd ElectronicsDiagnosticTool
   ```
EOF#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to create directories
create_directories() {
    echo "Creating directory structure..."
    mkdir -p diagnostics
    mkdir -p ai
    mkdir -p gui
    mkdir -p web_interface
    mkdir -p api
    mkdir -p integration
    mkdir -p security
    mkdir -p inventory
    mkdir -p tests
    mkdir -p config
    mkdir -p logs
    mkdir -p reports
    echo "Directories created successfully."
}

# Function to create Python files with content
create_python_files() {
    echo "Creating Python source files..."

    # diagnostics/diagnostics.py
    cat <<EOF > diagnostics/diagnostics.py
import psutil
import platform
import wmi
import random  # Placeholder for actual hardware data
from pySMART import Smart
import GPUtil
import serial
import logging

# Configure logging
logging.basicConfig(filename='../logs/diagnostics.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class CPUDiagnostics:
    def __init__(self):
        self.system = platform.system()

    def get_cpu_info(self):
        try:
            cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
            cpu_freq = psutil.cpu_freq()._asdict()
            cpu_temp = self.get_cpu_temperature()
            load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else "N/A"
            return {
                "CPU Usage (%)": cpu_usage,
                "CPU Frequency (MHz)": cpu_freq,
                "CPU Temperature (°C)": cpu_temp,
                "Load Average": load_avg
            }
        except Exception as e:
            logging.error(f"CPU Diagnostics Error: {e}")
            return {"Error": str(e)}

    def get_cpu_temperature(self):
        try:
            if self.system == "Windows":
                w = wmi.WMI(namespace="root\\wmi")
                temperature_info = w.MSAcpi_ThermalZoneTemperature()[0]
                return (temperature_info.CurrentTemperature / 10.0) - 273.15
            elif self.system == "Linux":
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp = float(f.read()) / 1000.0
                    return temp
            elif self.system == "Darwin":
                # macOS-specific temperature retrieval can be implemented using third-party tools or APIs
                return "N/A"
            else:
                return "N/A"
        except Exception as e:
            logging.error(f"CPU Temperature Retrieval Error: {e}")
            return f"Error: {e}"

class RAMDiagnostics:
    def get_memory_info(self):
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                "Total Memory (GB)": round(memory.total / (1024 ** 3), 2),
                "Available Memory (GB)": round(memory.available / (1024 ** 3), 2),
                "Memory Usage (%)": memory.percent,
                "Total Swap (GB)": round(swap.total / (1024 ** 3), 2),
                "Swap Usage (%)": swap.percent
            }
        except Exception as e:
            logging.error(f"RAM Diagnostics Error: {e}")
            return {"Error": str(e)}

class StorageDiagnostics:
    def get_storage_info(self):
        try:
            partitions = psutil.disk_partitions()
            storage_info = {}
            for partition in partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)._asdict()
                    io = psutil.disk_io_counters(perdisk=True)
                    disk_name = partition.device.split('/')[-1]
                    storage_info[partition.device] = {
                        "Mountpoint": partition.mountpoint,
                        "File System": partition.fstype,
                        "Usage": usage,
                        "IO Statistics": io.get(disk_name, "N/A")
                    }
                except PermissionError:
                    continue
            smart_info = self.get_smart_info()
            storage_info["SMART"] = smart_info
            return storage_info
        except Exception as e:
            logging.error(f"Storage Diagnostics Error: {e}")
            return {"Error": str(e)}

    def get_smart_info(self):
        try:
            drives = Smart.devices()
            smart_data = {}
            for drive in drives:
                smart_data[drive.name] = {
                    "Model": drive.model,
                    "Serial": drive.serial,
                    "Health": drive.assessment,
                    "Temperature": getattr(drive, 'temperature', 'N/A'),
                    "Reallocated Sectors": getattr(drive, 'reallocated_sector_count', 'N/A')
                }
            return smart_data
        except Exception as e:
            logging.error(f"SMART Data Retrieval Error: {e}")
            return f"Error retrieving SMART data: {e}"

class GPUDiagnostics:
    def get_gpu_info(self):
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = {}
            for gpu in gpus:
                gpu_info[gpu.id] = {
                    "Name": gpu.name,
                    "Load (%)": round(gpu.load * 100, 2),
                    "Free Memory (MB)": gpu.memoryFree,
                    "Used Memory (MB)": gpu.memoryUsed,
                    "Total Memory (MB)": gpu.memoryTotal,
                    "Temperature (°C)": gpu.temperature,
                    "UUID": gpu.uuid,
                    "Clock Speed (MHz)": gpu.clock,
                    "Fan Speed (%)": gpu.fan,
                }
            return gpu_info
        except Exception as e:
            logging.error(f"GPU Diagnostics Error: {e}")
            return {"Error": str(e)}

class BatteryDiagnostics:
    def get_battery_info(self):
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    "Battery Percent": battery.percent,
                    "Power Plugged In": battery.power_plugged,
                    "Battery Time Remaining (seconds)": battery.secsleft,
                    "Battery Health (%)": self.get_battery_health(),
                    "Charge Cycles": self.get_charge_cycles()
                }
            else:
                return {"Battery": "Not Available"}
        except Exception as e:
            logging.error(f"Battery Diagnostics Error: {e}")
            return {"Error": str(e)}

    def get_battery_health(self):
        # Implement actual battery health retrieval based on hardware APIs or external tools
        # Placeholder implementation
        return random.randint(70, 100)

    def get_charge_cycles(self):
        # Implement actual charge cycle retrieval based on hardware APIs or external tools
        # Placeholder implementation
        return random.randint(100, 500)

class PowerSupplyDiagnostics:
    def get_power_supply_info(self):
        try:
            # Placeholder for actual implementation
            # Requires interfacing with hardware via serial or other protocols
            voltage, stability = self.read_power_supply_voltage()
            return {
                "Power Supply Voltage (V)": voltage,
                "Voltage Stability": stability
            }
        except Exception as e:
            logging.error(f"Power Supply Diagnostics Error: {e}")
            return {"Error": str(e)}

    def read_power_supply_voltage(self):
        # Simulated voltage reading; replace with actual hardware communication
        # Example using pyserial to communicate with a multimeter or power supply unit
        try:
            ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Update COM port as needed
            ser.write(b'READ_VOLTS\n')  # Command depends on your device's protocol
            response = ser.readline().decode('utf-8').strip()
            ser.close()
            voltage = float(response)
            stability = self.assess_voltage_stability(voltage)
            return voltage, stability
        except Exception as e:
            logging.error(f"Power Supply Reading Error: {e}")
            return "N/A", "N/A"

    def assess_voltage_stability(self, voltage):
        # Placeholder for actual voltage stability assessment
        if 11.5 <= voltage <= 12.5:
            return "Stable"
        elif 10.0 <= voltage < 11.5 or 12.5 < voltage <= 13.5:
            return "Marginal"
        else:
            return "Unstable"
EOF

    # ai/ai_module.py
    cat <<EOF > ai/ai_module.py
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
EOF

    # ai/model_optimization.py
    cat <<EOF > ai/model_optimization.py
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

# Configure logging
logging.basicConfig(filename='../logs/model_optimization.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def optimize_model(X_train, y_train):
    try:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Best Score: {grid_search.best_score_}")
        return grid_search.best_estimator_
    except Exception as e:
        logging.error(f"Model Optimization Error: {e}")
        return None

if __name__ == "__main__":
    from ai_module import load_data, preprocess_data, train_model
    data = load_data('../data/diagnostic_data.csv')
    if data is not None:
        X, y = preprocess_data(data)
        if X is not None and y is not None:
            optimized_model = optimize_model(X, y)
            if optimized_model:
                joblib.dump(optimized_model, '../ai/optimized_failure_predictor.pkl')
                logging.info("Optimized model saved successfully.")
EOF

    # ai/model_explainability.py
    cat <<EOF > ai/model_explainability.py
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(filename='../logs/model_explainability.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def explain_model():
    try:
        model = joblib.load('../ai/optimized_failure_predictor.pkl')
        scaler = joblib.load('../ai/scaler.pkl')
        data = pd.read_csv('../data/diagnostic_data.csv').dropna()
        X = data[['Average_CPU_Usage', 'memory_usage', 'disk_health', 'gpu_temp', 'battery_health']]
        X_scaled = scaler.transform(X)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        # Summary Plot
        shap.summary_plot(shap_values, X, plot_type="bar")
        plt.savefig('../reports/shap_summary_plot.png')
        plt.clf()
        logging.info("SHAP summary plot generated and saved.")

        # Force Plot for the first prediction
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], X.iloc[0], matplotlib=True)
        plt.savefig('../reports/shap_force_plot.png')
        plt.clf()
        logging.info("SHAP force plot generated and saved.")
    except Exception as e:
        logging.error(f"Model Explainability Error: {e}")

if __name__ == "__main__":
    explain_model()
EOF

    # gui/gui.py
    cat <<EOF > gui/gui.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel, QMessageBox
)
import json
from workflow import execute_workflow

class DiagnosticApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Electronics Diagnostic Tool')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.status_label = QLabel('Status: Ready')
        layout.addWidget(self.status_label)

        self.run_button = QPushButton('Run Diagnostics')
        self.run_button.clicked.connect(self.run_diagnostics)
        layout.addWidget(self.run_button)

        self.report_area = QTextEdit()
        self.report_area.setReadOnly(True)
        layout.addWidget(self.report_area)

        self.setLayout(layout)

    def run_diagnostics(self):
        reply = QMessageBox.question(
            self, 'Confirm', 'Are you sure you want to run diagnostics?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.status_label.setText('Status: Running Diagnostics...')
            QApplication.processEvents()  # Update UI
            report = execute_workflow()
            if "Error" not in report:
                self.report_area.setText(json.dumps(report, indent=4))
                self.status_label.setText('Status: Diagnostics Completed')
                QMessageBox.information(self, 'Success', 'Diagnostics completed successfully.')
            else:
                self.report_area.setText(json.dumps(report, indent=4))
                self.status_label.setText('Status: Diagnostics Failed')
                QMessageBox.critical(self, 'Error', 'Diagnostics failed. Check logs for details.')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = DiagnosticApp()
    ex.show()
    sys.exit(app.exec_())
EOF

    # web_interface/web_interface.py
    cat <<EOF > web_interface/web_interface.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import json
from workflow import execute_workflow

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Electronics Diagnostic Dashboard"), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Run Diagnostics", id="run-button", color="primary"), width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Loading(id="loading", children=[html.Div(id="report-output")], type="default"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='cpu-usage'), width=6),
        dbc.Col(dcc.Graph(id='memory-usage'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='disk-usage'), width=6),
        dbc.Col(dcc.Graph(id='gpu-temp'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='battery-health'), width=6),
        dbc.Col(dcc.Graph(id='power-supply'), width=6)
    ])
], fluid=True)

@app.callback(
    Output("report-output", "children"),
    [
        Input("run-button", "n_clicks")
    ]
)
def run_diagnostics(n_clicks):
    if n_clicks:
        report = execute_workflow()
        if "Error" not in report:
            return html.Pre(json.dumps(report, indent=4))
        else:
            return html.Pre(json.dumps(report, indent=4))
    return ""

@app.callback(
    [
        Output('cpu-usage', 'figure'),
        Output('memory-usage', 'figure'),
        Output('disk-usage', 'figure'),
        Output('gpu-temp', 'figure'),
        Output('battery-health', 'figure'),
        Output('power-supply', 'figure')
    ],
    [Input("run-button", "n_clicks")]
)
def update_graphs(n_clicks):
    if n_clicks:
        report = execute_workflow()
        if "Error" not in report:
            cpu = report['Diagnostics']['CPU']['CPU Usage (%)']
            memory = report['Diagnostics']['Memory']['Memory Usage (%)']
            # Example: Extract disk read count
            disk_io = report['Diagnostics']['Storage']['IO Statistics']
            disk_read = disk_io['sda']['read_count'] if 'sda' in disk_io else 0
            gpu_temp = report['Diagnostics']['GPU']['0']['Temperature (°C)'] if '0' in report['Diagnostics']['GPU'] else 0
            battery = report['Diagnostics']['Battery']['Battery Health (%)'] if 'Battery Health (%)' in report['Diagnostics']['Battery'] else 100
            power = report['Diagnostics']['Power Supply']['Power Supply Voltage (V)'] if 'Power Supply Voltage (V)' in report['Diagnostics']['Power Supply'] else 0

            # Create figures
            cpu_fig = {
                'data': [{'x': list(range(len(cpu))), 'y': cpu, 'type': 'line', 'name': 'CPU Usage'}],
                'layout': {'title': 'CPU Usage (%)'}
            }
            memory_fig = {
                'data': [{'x': list(range(len(memory))), 'y': memory, 'type': 'line', 'name': 'Memory Usage'}],
                'layout': {'title': 'Memory Usage (%)'}
            }
            disk_fig = {
                'data': [{'x': ['Read Count'], 'y': [disk_read], 'type': 'bar', 'name': 'Disk Read Count'}],
                'layout': {'title': 'Disk Read Count'}
            }
            gpu_fig = {
                'data': [{'x': ['GPU Temperature'], 'y': [gpu_temp], 'type': 'bar', 'name': 'GPU Temp'}],
                'layout': {'title': 'GPU Temperature (°C)'}
            }
            battery_fig = {
                'data': [{'labels': ['Healthy', 'Needs Replacement'], 'values': [battery, 100 - battery], 'type': 'pie'}],
                'layout': {'title': 'Battery Health (%)'}
            }
            power_fig = {
                'data': [{'x': ['Voltage'], 'y': [power], 'type': 'bar', 'name': 'Power Supply Voltage'}],
                'layout': {'title': 'Power Supply Voltage (V)'}
            }

            return cpu_fig, memory_fig, disk_fig, gpu_fig, battery_fig, power_fig
    return {}, {}, {}, {}, {}, {}

if __name__ == '__main__':
    app.run_server(debug=True)
EOF

    # api/api.py
    cat <<EOF > api/api.py
 from flask import Flask, jsonify, request
 from workflow import execute_workflow
 import logging

 app = Flask(__name__)

 # Configure logging
 logging.basicConfig(filename='../logs/api.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 @app.route('/run_diagnostics', methods=['POST'])
 def run_diagnostics_api():
     try:
         report = execute_workflow()
         return jsonify(report), 200
     except Exception as e:
         logging.error(f"API Diagnostics Error: {e}")
         return jsonify({"Error": str(e)}), 500

 @app.route('/get_report/<report_id>', methods=['GET'])
 def get_report(report_id):
     # Implement logic to retrieve report by ID from the database or storage
     # Placeholder response
     return jsonify({"report_id": report_id, "data": "Report data here"}), 200

 if __name__ == '__main__':
     app.run(debug=True, port=5001)
EOF

    # integration/hardware_integration.py
    cat <<EOF > integration/hardware_integration.py
 import serial
 import logging

 # Configure logging
 logging.basicConfig(filename='../logs/hardware_integration.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 class Multimeter:
     def __init__(self, port='/dev/ttyUSB0', baudrate=9600, timeout=1):
         self.port = port
         self.baudrate = baudrate
         self.timeout = timeout

     def read_voltage(self):
         try:
             with serial.Serial(self.port, self.baudrate, timeout=self.timeout) as ser:
                 ser.write(b'READ_VOLTS\n')  # Command depends on your device's protocol
                 response = ser.readline().decode('utf-8').strip()
                 voltage = float(response)
                 logging.info(f"Voltage Read: {voltage} V")
                 return voltage
         except Exception as e:
             logging.error(f"Multimeter Read Voltage Error: {e}")
             return "N/A"

     def read_current(self):
         try:
             with serial.Serial(self.port, self.baudrate, timeout=self.timeout) as ser:
                 ser.write(b'READ_CURRENT\n')  # Command depends on your device's protocol
                 response = ser.readline().decode('utf-8').strip()
                 current = float(response)
                 logging.info(f"Current Read: {current} A")
                 return current
         except Exception as e:
             logging.error(f"Multimeter Read Current Error: {e}")
             return "N/A"
EOF

    # integration/cloud_storage.py
    cat <<EOF > integration/cloud_storage.py
 import boto3
 from botocore.exceptions import NoCredentialsError
 import logging

 # Configure logging
 logging.basicConfig(filename='../logs/cloud_storage.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 def upload_to_s3(file_name, bucket, object_name=None):
     s3_client = boto3.client('s3')
     try:
         s3_client.upload_file(file_name, bucket, object_name or file_name)
         logging.info(f"File {file_name} uploaded to {bucket} as {object_name or file_name}")
     except FileNotFoundError:
         logging.error(f"The file {file_name} was not found.")
     except NoCredentialsError:
         logging.error("Credentials not available.")
EOF

    # security/security.py
    cat <<EOF > security/security.py
 from cryptography.fernet import Fernet
 import os
 import logging

 # Configure logging
 logging.basicConfig(filename='../logs/security.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 def generate_key():
     key = Fernet.generate_key()
     with open("../config/secret.key", "wb") as key_file:
         key_file.write(key)
     logging.info("Encryption key generated and saved.")

 def load_key():
     try:
         return open("../config/secret.key", "rb").read()
     except Exception as e:
         logging.error(f"Load Key Error: {e}")
         return None

 def encrypt_data(data, key):
     try:
         f = Fernet(key)
         encrypted = f.encrypt(data.encode())
         logging.info("Data encrypted successfully.")
         return encrypted
     except Exception as e:
         logging.error(f"Encrypt Data Error: {e}")
         return None

 def decrypt_data(encrypted_data, key):
     try:
         f = Fernet(key)
         decrypted = f.decrypt(encrypted_data).decode()
         logging.info("Data decrypted successfully.")
         return decrypted
     except Exception as e:
         logging.error(f"Decrypt Data Error: {e}")
         return None

 if __name__ == "__main__":
     if not os.path.exists("../config/secret.key"):
         generate_key()
     key = load_key()
     sample_data = "Sensitive Diagnostic Report Data"
     encrypted = encrypt_data(sample_data, key)
     print(f"Encrypted: {encrypted}")
     decrypted = decrypt_data(encrypted, key)
     print(f"Decrypted: {decrypted}")
EOF

    # inventory/inventory_integration.py
    cat <<EOF > inventory/inventory_integration.py
 from sqlalchemy import create_engine, Column, Integer, String, Float
 from sqlalchemy.ext.declarative import declarative_base
 from sqlalchemy.orm import sessionmaker
 import logging

 # Configure logging
 logging.basicConfig(filename='../logs/inventory_integration.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 Base = declarative_base()

 class Part(Base):
     __tablename__ = 'parts'
     id = Column(Integer, primary_key=True)
     name = Column(String, unique=True, nullable=False)
     quantity = Column(Integer, nullable=False)
     cost = Column(Float, nullable=False)

 def update_inventory(part_name, quantity_used):
     try:
         engine = create_engine('postgresql://user:password@localhost/dbname')  # Update with your DB credentials
         Session = sessionmaker(bind=engine)
         session = Session()
         part = session.query(Part).filter_by(name=part_name).first()
         if part and part.quantity >= quantity_used:
             part.quantity -= quantity_used
             session.commit()
             logging.info(f"Updated inventory for {part_name}: -{quantity_used}")
         else:
             logging.warning(f"Insufficient inventory for {part_name}")
         session.close()
     except Exception as e:
         logging.error(f"Inventory Update Error: {e}")

 if __name__ == "__main__":
     update_inventory('RAM Module', 2)
EOF

    # tests/test_diagnostics.py
    cat <<EOF > tests/test_diagnostics.py
 import unittest
 from diagnostics.diagnostics import CPUDiagnostics, RAMDiagnostics, StorageDiagnostics, GPUDiagnostics, BatteryDiagnostics, PowerSupplyDiagnostics

 class TestDiagnostics(unittest.TestCase):

     def setUp(self):
         self.cpu = CPUDiagnostics()
         self.ram = RAMDiagnostics()
         self.storage = StorageDiagnostics()
         self.gpu = GPUDiagnostics()
         self.battery = BatteryDiagnostics()
         self.power = PowerSupplyDiagnostics()

     def test_cpu_info(self):
         cpu_info = self.cpu.get_cpu_info()
         self.assertIn("CPU Usage (%)", cpu_info)
         self.assertIn("CPU Frequency (MHz)", cpu_info)
         self.assertIn("CPU Temperature (°C)", cpu_info)

     def test_memory_info(self):
         memory_info = self.ram.get_memory_info()
         self.assertIn("Total Memory (GB)", memory_info)
         self.assertIn("Available Memory (GB)", memory_info)
         self.assertIn("Memory Usage (%)", memory_info)

     def test_storage_info(self):
         storage_info = self.storage.get_storage_info()
         self.assertIn("SMART", storage_info)

     def test_gpu_info(self):
         gpu_info = self.gpu.get_gpu_info()
         # Depending on system, GPU info may or may not be available
         # So, check if it's either a dictionary or an error message
         self.assertTrue(isinstance(gpu_info, dict) or "Error" in gpu_info)

     def test_battery_info(self):
         battery_info = self.battery.get_battery_info()
         # Battery info might not be available on desktops
         self.assertTrue(isinstance(battery_info, dict))

     def test_power_supply_info(self):
         power_info = self.power.get_power_supply_info()
         self.assertIn("Power Supply Voltage (V)", power_info)

 if __name__ == '__main__':
     unittest.main()
EOF

    # tests/test_integration.py
    cat <<EOF > tests/test_integration.py
 import unittest
 from workflow import run_all_diagnostics, generate_report, save_report, log_diagnostics, notify_technician

 class TestIntegration(unittest.TestCase):

     def test_full_diagnostics_workflow(self):
         report = run_all_diagnostics()
         self.assertIn("CPU", report["Diagnostics"])
         self.assertIn("Memory", report["Diagnostics"])
         self.assertIn("Storage", report["Diagnostics"])
         self.assertIn("GPU", report["Diagnostics"])
         self.assertIn("Battery", report["Diagnostics"])
         self.assertIn("Power Supply", report["Diagnostics"])
         self.assertIn("AI Prediction", report)

 if __name__ == '__main__':
     unittest.main()
EOF

    # tests/test_workflow.py
    cat <<EOF > tests/test_workflow.py
 import unittest
 from workflow import execute_workflow

 class TestWorkflow(unittest.TestCase):

     def test_execute_workflow(self):
         report = execute_workflow()
         self.assertIn("Timestamp", report)
         self.assertIn("Diagnostics", report)
         self.assertIn("AI Prediction", report)
         self.assertIn("CPU", report["Diagnostics"])
         self.assertIn("Memory", report["Diagnostics"])
         self.assertIn("Storage", report["Diagnostics"])

 if __name__ == '__main__':
     unittest.main()
EOF

    # workflow.py
    cat <<EOF > workflow.py
 from diagnostics.diagnostics import CPUDiagnostics, RAMDiagnostics, StorageDiagnostics, GPUDiagnostics, BatteryDiagnostics, PowerSupplyDiagnostics
 import json
 from datetime import datetime
 import logging
 from ai.ai_module import predict_failure
 from integration.cloud_storage import upload_to_s3
 from security.security import encrypt_data, load_key
 import smtplib
 from email.mime.text import MIMEText

 # Configure logging
 logging.basicConfig(filename='logs/workflow.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 def run_all_diagnostics():
     diagnostics = {}
     cpu = CPUDiagnostics()
     ram = RAMDiagnostics()
     storage = StorageDiagnostics()
     gpu = GPUDiagnostics()
     battery = BatteryDiagnostics()
     power = PowerSupplyDiagnostics()

     diagnostics["CPU"] = cpu.get_cpu_info()
     diagnostics["Memory"] = ram.get_memory_info()
     diagnostics["Storage"] = storage.get_storage_info()
     diagnostics["GPU"] = gpu.get_gpu_info()
     diagnostics["Battery"] = battery.get_battery_info()
     diagnostics["Power Supply"] = power.get_power_supply_info()

     return diagnostics

 def generate_report(diagnostics, prediction):
     report = {
         "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
         "Diagnostics": diagnostics,
         "AI Prediction": prediction
     }
     return report

 def save_report(report, filename="diagnostic_report.json"):
     try:
         with open(filename, "w") as file:
             json.dump(report, file, indent=4)
         logging.info(f"Report saved to {filename}")
     except Exception as e:
         logging.error(f"Error saving report: {e}")

 def log_diagnostics(report):
     try:
         with open('logs/diagnostic_logs.log', 'a') as f:
             f.write(json.dumps(report) + "\n")
         logging.info("Diagnostics logged successfully.")
     except Exception as e:
         logging.error(f"Error logging diagnostics: {e}")

 def notify_technician(report, technician_email="technician@example.com"):
     try:
         subject = f"Diagnostic Report - {report['Timestamp']}"
         body = json.dumps(report, indent=4)

         msg = MIMEText(body)
         msg['Subject'] = subject
         msg['From'] = "your_email@example.com"  # Replace with your email
         msg['To'] = technician_email

         with smtplib.SMTP_SSL('smtp.example.com', 465) as server:  # Replace with your SMTP server
             server.login("your_email@example.com", "your_password")  # Replace with your credentials
             server.sendmail("your_email@example.com", technician_email, msg.as_string())
         logging.info(f"Notification sent to {technician_email}")
     except Exception as e:
         logging.error(f"Error sending notification: {e}")

 def execute_workflow():
     try:
         # Step 1: Run Diagnostics
         diagnostics = run_all_diagnostics()
         logging.info("Diagnostics run successfully.")

         # Step 2: AI Prediction
         prediction_input = [
             diagnostics['CPU']['CPU Usage (%)'],
             diagnostics['Memory']['Memory Usage (%)'],
             diagnostics['Storage']['SMART']['sda']['Reallocated Sectors'] if 'sda' in diagnostics['Storage']['SMART'] else 0,
             diagnostics['GPU']['0']['Temperature (°C)'] if '0' in diagnostics['GPU'] else 0,
             diagnostics['Battery']['Battery Health (%)'] if 'Battery Health (%)' in diagnostics['Battery'] else 100
         ]
         prediction = predict_failure(*prediction_input)
         logging.info(f"AI Prediction: {prediction}")

         # Step 3: Generate Report
         report = generate_report(diagnostics, prediction)
         logging.info("Report generated successfully.")

         # Step 4: Save and Log Report
         save_report(report)
         log_diagnostics(report)
         logging.info("Report saved and logged successfully.")

         # Step 5: Upload to Cloud Storage
         upload_to_s3('diagnostic_report.json', 'your-s3-bucket-name')  # Replace with your S3 bucket
         logging.info("Report uploaded to cloud storage successfully.")

         # Step 6: Encrypt Report
         key = load_key()
         encrypted_report = encrypt_data(json.dumps(report), key)
         with open('diagnostic_report_encrypted.json', 'wb') as f:
             f.write(encrypted_report)
         logging.info("Report encrypted successfully.")

         # Step 7: Notify Technician
         notify_technician(report)
         logging.info("Technician notified successfully.")

         return report
     except Exception as e:
         logging.error(f"Workflow Execution Error: {e}")
         return {"Error": str(e)}

 if __name__ == "__main__":
     report = execute_workflow()
     print("Workflow Execution Completed:")
     print(report)
EOF

    # Dockerfile
    cat <<EOF > Dockerfile
 # Use official Python runtime as a parent image
 FROM python:3.9-slim

 # Set environment variables
 ENV PYTHONDONTWRITEBYTECODE 1
 ENV PYTHONUNBUFFERED 1

 # Set work directory
 WORKDIR /app

 # Install dependencies
 COPY requirements.txt .
 RUN pip install --upgrade pip
 RUN pip install --no-cache-dir -r requirements.txt

 # Copy project
 COPY . .

 # Expose port for Dash app
 EXPOSE 8050

 # Command to run the web interface
 CMD ["python", "web_interface/web_interface.py"]
EOF

    # requirements.txt
    cat <<EOF > requirements.txt
 psutil
 pySMART
 py-cpuinfo
 wmi; platform_system == "Windows"
 GPUtil
 pyserial
 dash
 dash-bootstrap-components
 Flask
 SQLAlchemy
 Flask-Bcrypt
 Flask-JWT-Extended
 scikit-learn
 pandas
 numpy
 joblib
 shap
 PyQt5
 boto3
 cryptography
EOF

    # README.md
    cat <<EOF > README.md
 # Electronics Diagnostic Tool

 ## Overview

 A comprehensive diagnostic software for electronics testing and refurbishment, integrating advanced AI for predictive analytics.

 ## Directory Structure

 - `diagnostics/` - Hardware diagnostic modules
 - `ai/` - AI and machine learning modules
 - `gui/` - Desktop GUI application
 - `web_interface/` - Web-based dashboard
 - `api/` - RESTful API services
 - `integration/` - External tool integrations
 - `security/` - Data encryption and security modules
 - `inventory/` - Inventory management integrations
 - `tests/` - Unit and integration tests
 - `config/` - Configuration files
 - `logs/` - Log files
 - `reports/` - Generated reports
 - `Dockerfile` - Docker configuration for containerization
 - `requirements.txt` - Python dependencies

 ## Setup Instructions

 1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/ElectronicsDiagnosticTool.git
    cd ElectronicsDiagnosticTool
    ```
 EOF

    echo "Python source files created successfully."
}

# Function to create other necessary scripts
create_other_scripts() {
    echo "Creating additional scripts..."

    # backup.py
    cat <<EOF > backup.py
 import boto3
 import os
 import datetime
 import logging

 # Configure logging
 logging.basicConfig(filename='logs/backup.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 def backup_database():
     try:
         # Assuming PostgreSQL
         os.system("pg_dump -U your_username -h localhost your_dbname > backup.sql")
         logging.info("Database backup created successfully.")
     except Exception as e:
         logging.error(f"Database Backup Error: {e}")

 def upload_backup_to_s3(file_name, bucket):
     s3_client = boto3.client('s3')
     try:
         s3_client.upload_file(file_name, bucket, file_name)
         logging.info(f"Backup {file_name} uploaded to {bucket}")
     except Exception as e:
         logging.error(f"S3 Upload Error: {e}")

 if __name__ == "__main__":
     backup_database()
     backup_file = f"backup_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.sql"
     os.rename("backup.sql", backup_file)
     upload_backup_to_s3(backup_file, 'your-s3-bucket-name')  # Replace with your S3 bucket
EOF

    # report_generation.py
    cat <<EOF > report_generation.py
 from reportlab.lib.pagesizes import letter
 from reportlab.pdfgen import canvas
 import json
 import logging

 # Configure logging
 logging.basicConfig(filename='logs/report_generation.log', level=logging.INFO,
                     format='%(asctime)s:%(levelname)s:%(message)s')

 def generate_pdf_report(report, filename="diagnostic_report.pdf"):
     try:
         c = canvas.Canvas(filename, pagesize=letter)
         width, height = letter
         c.setFont("Helvetica", 12)
         y = height - 50
         c.drawString(50, y, f"Diagnostic Report - {report['Timestamp']}")
         y -= 30
         for section, details in report['Diagnostics'].items():
             c.drawString(50, y, f"{section}:")
             y -= 20
             for key, value in details.items():
                 c.drawString(70, y, f"{key}: {value}")
                 y -= 15
             y -= 10
         c.drawString(50, y, f"AI Prediction: {report['AI Prediction']['Prediction']} with confidence {report['AI Prediction']['Confidence']}%")
         c.save()
         logging.info(f"PDF report generated: {filename}")
     except Exception as e:
         logging.error(f"PDF Report Generation Error: {e}")

 if __name__ == "__main__":
     sample_report = {
         "Timestamp": "2024-04-27 14:30:45",
         "Diagnostics": {
             "CPU": {
                 "CPU Usage (%)": [45.0, 50.0, 55.0],
                 "CPU Frequency (MHz)": {"current": 2400, "min": 800, "max": 2400},
                 "CPU Temperature (°C)": 65.0
             },
             "Memory": {
                 "Total Memory (GB)": 16.0,
                 "Available Memory (GB)": 8.0,
                 "Memory Usage (%)": 50.0
             },
             # Additional sections...
         },
         "AI Prediction": {
             "Prediction": "Healthy",
             "Confidence": 95.0
         }
     }
     generate_pdf_report(sample_report)
EOF

    echo "Additional scripts created successfully."
}

# Function to create virtual environment and install dependencies
setup_virtualenv() {
    echo "Setting up Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Virtual environment setup and dependencies installed successfully."
}

# Function to initialize the database (example with SQLAlchemy)
initialize_database() {
    echo "Initializing the database..."
    # Example: Create tables for inventory
    python -c "
from sqlalchemy import create_engine
from inventory.inventory_integration import Base

engine = create_engine('postgresql://user:password@localhost/dbname')  # Update with your DB credentials
Base.metadata.create_all(engine)
print('Database initialized successfully.')
"
    echo "Database initialized."
}

# Main execution
echo "Starting setup process..."
create_directories
create_python_files
create_other_scripts
setup_virtualenv
initialize_database
echo "Setup completed successfully."

# Optional: Instructions for user
echo "------------------------------------------------------------"
echo "Setup Summary:"
echo "- Directory structure created."
echo "- Source files generated in respective directories."
echo "- Python virtual environment 'venv' created and dependencies installed."
echo "- Database initialized (ensure PostgreSQL is running and credentials are correct)."
echo "------------------------------------------------------------"
echo "To activate the virtual environment, run: source venv/bin/activate"
echo "To run the desktop GUI: python gui/gui.py"
echo "To run the web interface: python web_interface/web_interface.py"
echo "To run the API server: python api/api.py"
echo "To execute the workflow: python workflow.py"
echo "------------------------------------------------------------"
