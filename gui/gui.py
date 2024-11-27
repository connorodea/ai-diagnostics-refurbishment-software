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
