#!/bin/bash
# Deploy gas-meter-ocr on Raspberry Pi Zero 2W (Trixie)
# Usage: bash scripts/deploy_rpi.sh [project_dir]

set -euo pipefail

PROJECT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_VERSION="3.11"
SWAP_SIZE="1G"
SWAP_FILE="/swapfile"

echo "=== Deploying gas-meter-ocr to RPi Zero 2W ==="
echo "Project directory: ${PROJECT_DIR}"

# --- System packages ---
echo ""
echo "=== Installing system dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    libcap-dev \
    libcamera-dev \
    python3-libcamera \
    python3-kms++ \
    htop

# --- Swap (needed for building C extensions on 512MB RAM) ---
echo ""
echo "=== Configuring swap ==="
if [ -f "${SWAP_FILE}" ]; then
    echo "Swap file already exists, skipping"
else
    echo "Creating ${SWAP_SIZE} swap file"
    sudo fallocate -l "${SWAP_SIZE}" "${SWAP_FILE}"
    sudo chmod 600 "${SWAP_FILE}"
    sudo mkswap "${SWAP_FILE}"
    sudo swapon "${SWAP_FILE}"
    echo "Swap enabled"
fi
swapon --show

# --- Virtual environment ---
echo ""
echo "=== Setting up Python ${PYTHON_VERSION} virtual environment ==="
cd "${PROJECT_DIR}"

if [ -d ".venv" ]; then
    echo "Removing existing .venv"
    rm -rf .venv
fi

python${PYTHON_VERSION} -m venv .venv --system-site-packages
source .venv/bin/activate

echo "Python: $(python --version)"
echo "Pip: $(pip --version)"

# --- Install project ---
echo ""
echo "=== Installing project with RPi extras ==="
TMPDIR="${PROJECT_DIR}/pip-tmp" pip install -e ".[rpi]"
rm -rf "${PROJECT_DIR}/pip-tmp"

# --- Verify ---
echo ""
echo "=== Verifying installation ==="
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import flask; print(f'Flask: {flask.__version__}')"
python -c "import RPi.GPIO; print('RPi.GPIO: OK')"
python -c "import libcamera; print('libcamera: OK')"
python -c "from picamera2 import Picamera2; print('picamera2: OK')"

echo ""
echo "=== Deployment complete ==="
echo "Activate with: source ${PROJECT_DIR}/.venv/bin/activate"
