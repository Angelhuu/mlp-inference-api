#!/bin/bash
pip install -r requirements.txt
gunicorn -w 1 -k uvicorn.workers.UvicornWorker api:app --bind=0.0.0.0:8000
