#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# if you don't have postgresql downloaded, then uncomment this and download it with brew or sudo apt on unix
# echo "Installing PostgreSQL..."
# sudo apt-get update
# sudo apt-get install -y postgresql postgresql-contrib

# Step 1: Start PostgreSQL service
# echo "Starting PostgreSQL service..."
# sudo systemctl start postgresql
# sudo systemctl enable postgresql

# Step 3: Create database, user, and grant privileges
echo "Setting up PostgreSQL database..."
sudo -i -u postgres psql <<EOF
CREATE USER diagnostic_user WITH PASSWORD 'securepassword123';
CREATE DATABASE diagnostic_db;
GRANT ALL PRIVILEGES ON DATABASE diagnostic_db TO diagnostic_user;
EOF

# Step 4: Create .env file inside the project directory
echo "Creating .env file..."
cat <<EOF > .env
DB_NAME=diagnostic_db
DB_USER=diagnostic_user
DB_PASSWORD=securepassword123
DB_HOST=localhost
DB_PORT=5432
EOF

# Step 5: Confirm setup
echo "Database setup complete!"
echo "Credentials saved to .env file:"
cat .env
