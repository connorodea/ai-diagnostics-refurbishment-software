
# 🔧 AI Diagnostics and Refurbishment Software

**A cutting-edge AI-powered diagnostic platform for testing and refurbishing electronic devices.**  
This project combines advanced AI predictions, hardware diagnostics, inventory management, and seamless integrations to streamline the refurbishment process.

---

## 🌟 Features

- **Comprehensive Diagnostics**:
  - Real-time CPU, RAM, storage, GPU, and battery health analysis.
  - Power supply voltage and stability monitoring.
- **AI-Driven Predictions**:
  - Detect potential hardware failures using machine learning models.
- **Multi-Platform Interfaces**:
  - **Desktop GUI**: Intuitive and user-friendly local application.
  - **Web Dashboard**: Remote access for management and reporting.
  - **RESTful API**: Flexible integration with external tools and services.
- **Inventory Integration**:
  - Track component usage and automate inventory updates.
- **Secure Data Handling**:
  - Encrypt sensitive data and integrate secure cloud storage options.
- **Scalable Architecture**:
  - Dockerized deployment for rapid scaling and portability.

---

## 📂 Project Structure

```plaintext
.
├── Dockerfile               # Docker container configuration
├── README.md                # Project documentation
├── ai                       # AI modules for predictions and optimizations
│   ├── ai_module.py
│   ├── model_explainability.py
│   └── model_optimization.py
├── api                      # RESTful API implementation
│   └── api.py
├── config                   # Configuration files
├── diagnostics              # Hardware diagnostics modules
│   └── diagnostics.py
├── gui                      # Desktop GUI implementation
│   └── gui.py
├── integration              # Cloud storage and hardware integration
│   ├── cloud_storage.py
│   └── hardware_integration.py
├── inventory                # Inventory management integrations
│   └── inventory_integration.py
├── logs                     # Log files for diagnostics and debugging
├── tests                    # Unit and integration test modules
│   ├── test_diagnostics.py
│   ├── test_integration.py
│   └── test_workflow.py
├── workflow.py              # Core workflow orchestration
├── setup.sh                 # Script to initialize the project structure
├── setup_db.sh              # Script to set up PostgreSQL database
├── requirements.txt         # Python dependencies
└── verify_db.sh             # Script to verify database setup
```

---

## 🛠️ Setup Instructions

### Prerequisites

- **Python**: Version 3.9 or higher.
- **PostgreSQL**: Installed and running.
- **Docker**: For containerized deployment (optional).

---

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/connorodea/ai-diagnostics-refurbishment-software.git
   cd ai-diagnostics-refurbishment-software
   ```

2. **Run the Setup Script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Set Up the Database**:
   ```bash
   chmod +x setup_db.sh
   ./setup_db.sh
   ```

4. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ How to Use

### 1. **Run the Core Workflow**
   Execute the diagnostic process:
   ```bash
   python workflow.py
   ```

### 2. **Launch the Desktop GUI**
   Start the local graphical interface:
   ```bash
   python gui/gui.py
   ```

### 3. **Access the Web Dashboard**
   Start the Dash-based web interface:
   ```bash
   python web_interface/web_interface.py
   ```

### 4. **Run the API Server**
   Launch the RESTful API for integration:
   ```bash
   python api/api.py
   ```

---

## 🔬 Testing

Run all unit and integration tests:
```bash
python -m unittest discover tests
```

---

## 🚀 Deployment

### Docker Deployment (Optional)

1. **Build the Docker Image**:
   ```bash
   docker build -t ai-diagnostics-tool .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8000:8000 ai-diagnostics-tool
   ```

---

## 🌐 Integrations

- **Cloud Storage**:
  - Automatically upload diagnostic reports to AWS S3 with `integration/cloud_storage.py`.
- **Database**:
  - Store diagnostic results and inventory data in PostgreSQL for secure and scalable storage.

---

## 🛡️ Security Features

- **Encryption**:
  - Sensitive data is encrypted with the `security/security.py` module.
- **Authentication**:
  - API endpoints support token-based authentication (JWT, optional).

---

## 🔧 Maintenance and Backup

### Database Backup
Run the following command to back up the PostgreSQL database:
```bash
pg_dump -U postgres -d diagnostic_db > diagnostic_db_backup.sql
```

### Restore Database
Restore the database from a backup file:
```bash
psql -U postgres -d diagnostic_db < diagnostic_db_backup.sql
```

---

## 📝 Roadmap

### Planned Enhancements
- **Real-Time Monitoring**:
  - AI-powered diagnostics with live status updates.
- **Mobile Integration**:
  - Mobile app support for on-the-go diagnostics.
- **Expanded Hardware Support**:
  - Add testing capabilities for additional components and peripherals.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👥 Contributors

- **Connor O'Dea** - Project Lead  
  GitHub: [connorodea](https://github.com/connorodea)

---

## 💡 Feedback and Contributions

We welcome your feedback and contributions! Please open an issue or submit a pull request to improve the project.