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
