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
