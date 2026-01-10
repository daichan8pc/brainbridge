#!/bin/bash
exec > /home/pi/brainbridge_startup.log 2>&1

echo "--- Startup Script Initiated ---"

export DISPLAY=:0

echo "Starting Streamlit..."
source /home/pi/BrainBridge/app/venv/bin/activate
cd /home/pi/BrainBridge/app

OPENBLAS_CORETYPE=ARMV8 streamlit run main.py --server.port 8501 --server.headless true &

echo "Waiting for server..."
sleep 20

echo "Starting Chromium..."
chromium-browser --kiosk --incognito http://localhost:8501