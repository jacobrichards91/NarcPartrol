#!/usr/bin/env bash
# NarcPartrol — Linux/macOS convenience wrapper
# The real installer is install.py (works on Windows, Linux, and macOS).
set -euo pipefail
exec python3 "$(dirname "$0")/install.py" "$@"
