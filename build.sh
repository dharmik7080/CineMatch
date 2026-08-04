#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Run migrations to update PostgreSQL schema in production
python manage.py migrate --no-input

# Load static files
python manage.py collectstatic --no-input

# Load pre-computed recommendation index fixtures into the database
python manage.py loaddata recommendations.json
