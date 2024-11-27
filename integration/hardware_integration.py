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
