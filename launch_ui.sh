#!/usr/bin/env bash
# NarcPartrol UI launcher (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")"

echo "Starting NarcPartrol UI..."
echo "Your browser will open to http://localhost:8501"
echo "Press Ctrl+C to stop the server."
echo

exec python3 -m streamlit run app.py \
    --server.headless=false \
    --browser.gatherUsageStats=false
