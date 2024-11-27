#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Load credentials from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "Error: .env file not found in the current directory."
  exit 1
fi

# Check if PostgreSQL is running
echo "Checking if PostgreSQL service is running..."
if ! systemctl is-active --quiet postgresql; then
  echo "PostgreSQL service is not running. Starting it now..."
  sudo systemctl start postgresql
fi

# Verify the database and user
echo "Verifying PostgreSQL database setup..."
psql_command="psql -U $DB_USER -h $DB_HOST -p $DB_PORT -d $DB_NAME -c"

echo "Listing all databases to confirm 'diagnostic_db' exists..."
$psql_command "\l"

echo "Checking privileges of user '$DB_USER' on database '$DB_NAME'..."
$psql_command "SELECT grantee, privilege_type FROM information_schema.role_table_grants WHERE table_schema = 'public';"

echo "PostgreSQL database setup verification complete!"
