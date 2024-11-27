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
                w = wmi.WMI(namespace="root\wmi")
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
