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
