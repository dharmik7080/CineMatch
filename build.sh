#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Run migrations to update PostgreSQL schema in production
python cinematch_project/manage.py migrate --no-input

# Load static files
python cinematch_project/manage.py collectstatic --no-input

# Load pre-computed recommendation index fixtures into the database
python cinematch_project/manage.py loaddata cinematch_project/recommendations.json
