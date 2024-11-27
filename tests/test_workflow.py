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
