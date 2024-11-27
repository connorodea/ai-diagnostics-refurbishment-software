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
