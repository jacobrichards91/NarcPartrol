#!/usr/bin/env bash
# NarcPartrol UI launcher (Linux / macOS)
# Finds whichever Python has streamlit installed.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for cmd in python3.12 python3.11 python3.13 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1 && "$cmd" -c "import streamlit" >/dev/null 2>&1; then
        PY="$cmd"
        break
    fi
done

if [[ -z "$PY" ]]; then
    echo "ERROR: Could not find a Python with streamlit installed."
    echo "Run the installer first:"
    echo "    python3.12 install.py   # or whichever python you want to use"
    exit 1
fi

echo "Starting NarcPartrol UI with $PY ..."
echo "Browser will open to http://localhost:8501"
echo "Press Ctrl+C to stop the server."
echo

exec "$PY" -m streamlit run app.py \
    --server.headless=false \
    --browser.gatherUsageStats=false
