#!/bin/bash
set -e  # Stop script if any command fails

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Gunicorn server..."
gunicorn -w 1 -k uvicorn.workers.UvicornWorker api:app --bind=0.0.0.0:8000
