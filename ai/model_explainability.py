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
