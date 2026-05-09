#!/usr/bin/env bash
# =============================================================================
# NarcPartrol Installer
# Sets up all system and Python dependencies for the HOA snapshot pipeline.
# Tested on Ubuntu 20.04/22.04/24.04 with an NVIDIA GPU.
# =============================================================================
set -euo pipefail

# --- Colour helpers ----------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}  ✓  $*${NC}"; }
info() { echo -e "${CYAN}  →  $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠  $*${NC}"; }
fail() { echo -e "${RED}  ✗  $*${NC}" >&2; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}━━━  $*  ━━━${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DEST="$HOME/Documents/NarcPartrol"
CONFIG_FILE="$CONFIG_DEST/narcpartrol.toml"

echo -e "${BOLD}"
echo "  ███╗   ██╗ █████╗ ██████╗  ██████╗"
echo "  ████╗  ██║██╔══██╗██╔══██╗██╔════╝"
echo "  ██╔██╗ ██║███████║██████╔╝██║"
echo "  ██║╚██╗██║██╔══██║██╔══██╗██║"
echo "  ██║ ╚████║██║  ██║██║  ██║╚██████╗"
echo "  ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  PARTROL"
echo -e "${NC}"
echo "  HOA Compliance Snapshot Pipeline — Installer"
echo "  ─────────────────────────────────────────────"

# =============================================================================
# Step 1 — Python version
# =============================================================================
header "Step 1 / 7 — Python"

PYTHON=$(command -v python3 || true)
[[ -z "$PYTHON" ]] && fail "python3 not found. Install Python 3.11+ first."

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ $PY_MAJOR -lt 3 || ($PY_MAJOR -eq 3 && $PY_MINOR -lt 11) ]]; then
    fail "Python 3.11+ required (found $PY_VERSION). Install it and re-run."
fi
ok "Python $PY_VERSION"

# =============================================================================
# Step 2 — GPU / CUDA detection
# =============================================================================
header "Step 2 / 7 — GPU & CUDA"

GPU_FOUND=false
CUDA_VERSION=""
TORCH_CUDA_TAG="cpu"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[\d.]+" 2>/dev/null || true)

    if [[ -n "$GPU_NAME" ]]; then
        GPU_FOUND=true
        ok "GPU:  $GPU_NAME"
        ok "CUDA: $CUDA_VERSION"

        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        if [[ "$CUDA_MAJOR" -ge 12 ]]; then
            TORCH_CUDA_TAG="cu121"
        elif [[ "$CUDA_MAJOR" -eq 11 ]]; then
            TORCH_CUDA_TAG="cu118"
        else
            warn "CUDA $CUDA_VERSION is older than 11.  Falling back to CPU PyTorch."
            warn "YOLOv8 will run on CPU and will be very slow."
            TORCH_CUDA_TAG="cpu"
        fi
    fi
fi

if [[ "$GPU_FOUND" == "false" ]]; then
    warn "No NVIDIA GPU detected (nvidia-smi not found or returned nothing)."
    warn "The pipeline will run on CPU.  Expect significantly longer processing times."
    warn "If you have a GPU, make sure the NVIDIA driver is installed and try again."
fi

# =============================================================================
# Step 3 — System packages (ffmpeg + exiftool)
# =============================================================================
header "Step 3 / 7 — System packages"

install_apt_pkg() {
    local pkg="$1"
    if dpkg -s "$pkg" &>/dev/null; then
        ok "$pkg already installed"
    else
        info "Installing $pkg ..."
        sudo apt-get install -y "$pkg"
        ok "$pkg installed"
    fi
}

if ! command -v apt-get &>/dev/null; then
    warn "apt-get not found — skipping system package installation."
    warn "Make sure ffmpeg and exiftool are installed manually:"
    warn "  ffmpeg  :  https://ffmpeg.org/download.html"
    warn "  exiftool:  https://exiftool.org/"
else
    sudo apt-get update -qq
    install_apt_pkg ffmpeg
    install_apt_pkg libimage-exiftool-perl
fi

# Verify the tools are usable
command -v ffmpeg    &>/dev/null && ok "ffmpeg  $(ffmpeg -version 2>&1 | head -1 | grep -oP 'version \K\S+')" \
    || warn "ffmpeg not in PATH — frame extraction will fail."
command -v exiftool  &>/dev/null && ok "exiftool $(exiftool -ver)" \
    || warn "exiftool not in PATH — GPS track extraction will fall back to ffprobe."

# =============================================================================
# Step 4 — PyTorch (CUDA-enabled if GPU found)
# =============================================================================
header "Step 4 / 7 — PyTorch"

# Check if torch is already installed with the right CUDA support
TORCH_OK=false
if "$PYTHON" -c "import torch" &>/dev/null; then
    TORCH_CUDA=$("$PYTHON" -c "import torch; print(torch.cuda.is_available())")
    TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
    if [[ "$GPU_FOUND" == "true" && "$TORCH_CUDA" == "True" ]]; then
        ok "PyTorch $TORCH_VER already installed with CUDA support"
        TORCH_OK=true
    elif [[ "$GPU_FOUND" == "false" && "$TORCH_CUDA" == "False" ]]; then
        ok "PyTorch $TORCH_VER already installed (CPU mode)"
        TORCH_OK=true
    else
        warn "PyTorch $TORCH_VER found but CUDA availability = $TORCH_CUDA (GPU_FOUND=$GPU_FOUND)"
        warn "Reinstalling with correct CUDA support..."
    fi
fi

if [[ "$TORCH_OK" == "false" ]]; then
    if [[ "$TORCH_CUDA_TAG" == "cpu" ]]; then
        info "Installing PyTorch (CPU) ..."
        "$PYTHON" -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
    else
        info "Installing PyTorch with CUDA ($TORCH_CUDA_TAG) ..."
        "$PYTHON" -m pip install --upgrade torch torchvision \
            --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
    fi

    TORCH_CUDA=$("$PYTHON" -c "import torch; print(torch.cuda.is_available())")
    TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
    ok "PyTorch $TORCH_VER installed  (cuda_available=$TORCH_CUDA)"
fi

# =============================================================================
# Step 5 — Python packages
# =============================================================================
header "Step 5 / 7 — Python packages"

info "Installing requirements from requirements.txt ..."
# torch is already installed above; pip won't downgrade it when ultralytics pulls it
"$PYTHON" -m pip install --upgrade pip --quiet
"$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt"
ok "All Python packages installed"

# =============================================================================
# Step 6 — Pre-download YOLOv8x-oiv7 model weights
# =============================================================================
header "Step 6 / 7 — YOLO model download"

MODEL_NAME=$(grep '^model' "$SCRIPT_DIR/narcpartrol.toml" 2>/dev/null \
    | head -1 | grep -oP '".*?"' | tr -d '"' || echo "yolov8x-oiv7.pt")

info "Pre-downloading $MODEL_NAME (~140 MB — this happens once) ..."
"$PYTHON" - <<PYEOF
from ultralytics import YOLO
import sys
try:
    YOLO("${MODEL_NAME}")
    print("  Model ready.")
except Exception as e:
    print(f"  Warning: could not pre-download model: {e}", file=sys.stderr)
    print("  It will be downloaded automatically on the first pipeline run.")
PYEOF
ok "Model weights ready"

# =============================================================================
# Step 7 — User config file
# =============================================================================
header "Step 7 / 7 — User config"

mkdir -p "$CONFIG_DEST"

if [[ -f "$CONFIG_FILE" ]]; then
    ok "Config already exists at $CONFIG_FILE — leaving it unchanged."
    info "(Delete it and re-run the installer to reset to defaults.)"
else
    cp "$SCRIPT_DIR/narcpartrol.toml" "$CONFIG_FILE"
    ok "Config installed at $CONFIG_FILE"
fi

# =============================================================================
# Final verification
# =============================================================================
header "Verification"

"$PYTHON" - <<PYEOF
import sys, pathlib
sys.path.insert(0, "${SCRIPT_DIR}")

errors = []

# Config loader
try:
    import config
    path = config.config_file_path()
    print(f"  config  ✓  loaded from {path or 'built-in defaults'}")
except Exception as e:
    errors.append(f"config: {e}")

# Core pipeline stages (import only — no GPU needed)
for mod in ["stages.ingest","stages.segment","stages.sampler",
            "stages.scorer","stages.exporter"]:
    try:
        __import__(mod)
        print(f"  {mod.split('.')[-1]:<12}✓")
    except Exception as e:
        errors.append(f"{mod}: {e}")

# PyTorch + CUDA
try:
    import torch
    cuda = torch.cuda.is_available()
    dev  = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"  torch        ✓  {torch.__version__}  device={dev}")
except Exception as e:
    errors.append(f"torch: {e}")

# OpenCV
try:
    import cv2
    print(f"  opencv       ✓  {cv2.__version__}")
except Exception as e:
    errors.append(f"opencv: {e}")

# EasyOCR (import only)
try:
    import easyocr  # noqa
    print(f"  easyocr      ✓")
except Exception as e:
    errors.append(f"easyocr: {e}")

if errors:
    print("\n  Issues found:", file=sys.stderr)
    for err in errors:
        print(f"    ✗  {err}", file=sys.stderr)
    sys.exit(1)
else:
    print("\n  All checks passed.")
PYEOF

# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  NarcPartrol is ready.${NC}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Config file:  $CONFIG_FILE"
echo "  Edit it to tune lot size, scoring weights, and more."
echo ""
echo "  Run the pipeline:"
echo "    python3 ${SCRIPT_DIR}/pipeline.py drive.mov --output ./results"
echo "    python3 ${SCRIPT_DIR}/pipeline.py *.mov    --output ./results --skip-cloud"
echo ""
echo "  Options:"
echo "    --min-frontage METERS   Override GPS lot-size threshold"
echo "    --fps N                 Override frame sampling rate"
echo "    --skip-cloud            Disable Claude Vision gate (offline mode)"
echo ""
