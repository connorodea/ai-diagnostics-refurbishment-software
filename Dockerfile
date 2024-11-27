 # Use official Python runtime as a parent image
 FROM python:3.9-slim

 # Set environment variables
 ENV PYTHONDONTWRITEBYTECODE 1
 ENV PYTHONUNBUFFERED 1

 # Set work directory
 WORKDIR /app

 # Install dependencies
 COPY requirements.txt .
 RUN pip install --upgrade pip
 RUN pip install --no-cache-dir -r requirements.txt

 # Copy project
 COPY . .

 # Expose port for Dash app
 EXPOSE 8050

 # Command to run the web interface
 CMD ["python", "web_interface/web_interface.py"]
