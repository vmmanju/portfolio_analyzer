#!/bin/bash
set -e

# 1. Start FastAPI Assistant Backend in the background
# We'll run it on port 8000. 
# Cloud Run normally listens on 8080, which we'll give to Streamlit.
# However, for both to work, we usually need a proxy.
# For simplicity, if the user only gets one port (8080), 
# we can have FastAPI listen on 8080 and have IT serve or redirect to streamlit,
# OR we can run Streamlit on 8080 and have it call the API.
# Given the user's gcloud command uses --port 8080, that is our entry point.

echo "Starting FastAPI Assistant on port 8000..."
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit Dashboard on port 8080..."
# In Cloud Run, the application MUST listen on $PORT (which is usually 8080)
streamlit run dashboard/run.py \
    --server.port=${PORT:-8080} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
