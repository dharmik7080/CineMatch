#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Run migrations to update PostgreSQL schema in production
python manage.py migrate --no-input

# Create superuser automatically if environment variables are set
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
    echo "Creating superuser automatically..."
    python manage.py createsuperuser --no-input || echo "Superuser already exists or could not be created."
fi

# Load static files
python manage.py collectstatic --no-input
