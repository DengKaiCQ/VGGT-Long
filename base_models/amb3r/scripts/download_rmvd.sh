#!/bin/bash
# Download and preprocess datasets for Robust Multi-View Depth (RMVD) benchmark.
# Datasets: ETH3D, DTU, ScanNet, Tanks and Temples
#
# Usage (from project root):
#   chmod +x scripts/download_rmvd.sh
#   bash scripts/download_rmvd.sh
#
# ScanNet requires a manual download step — see instructions below.

set -e

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${PROJECT_ROOT}/data/rmvd"
RMVD_SCRIPTS="${PROJECT_ROOT}/benchmark/tools/robustmvd/rmvd/data/scripts"

mkdir -p "$DATA_ROOT"

echo "============================================"
echo "  RMVD Data Download & Preprocessing"
echo "  Target directory: $DATA_ROOT"
echo "============================================"

# ──────────────────────────────────────────────
# 1. ETH3D
# ──────────────────────────────────────────────
echo ""
echo ">>> [1/4] ETH3D"
ETH3D_DIR="${DATA_ROOT}/eth3d"
if [ -d "$ETH3D_DIR" ] && [ "$(ls -A "$ETH3D_DIR" 2>/dev/null)" ]; then
    echo "  ETH3D already exists at $ETH3D_DIR, skipping."
else
    echo "  Downloading ETH3D..."
    bash "$RMVD_SCRIPTS/download_eth3d.sh" "$ETH3D_DIR"
    echo "  ETH3D download complete."
fi

# ──────────────────────────────────────────────
# 2. DTU (download + convert)
# ──────────────────────────────────────────────
echo ""
echo ">>> [2/4] DTU"
DTU_DIR="${DATA_ROOT}/dtu"
DTU_RAW_DIR="${DATA_ROOT}/dtu_raw"
if [ -d "$DTU_DIR" ] && [ "$(ls -A "$DTU_DIR" 2>/dev/null)" ]; then
    echo "  DTU already exists at $DTU_DIR, skipping."
else
    echo "  Downloading DTU raw data..."
    bash "$RMVD_SCRIPTS/download_dtu.sh" "$DTU_RAW_DIR"

    echo "  Converting DTU to required format..."
    python "$RMVD_SCRIPTS/convert_dtu.py" "$DTU_RAW_DIR" "$DTU_DIR"

    echo "  Cleaning up DTU raw data..."
    rm -rf "$DTU_RAW_DIR"
    echo "  DTU download and conversion complete."
fi

# ──────────────────────────────────────────────
# 3. ScanNet (requires manual download)
# ──────────────────────────────────────────────
echo ""
echo ">>> [3/4] ScanNet"
SCANNET_DIR="${DATA_ROOT}/scannet"
SCANNET_ORIG_DIR="${DATA_ROOT}/scannet_orig"
if [ -d "$SCANNET_DIR" ] && [ "$(ls -A "$SCANNET_DIR" 2>/dev/null)" ]; then
    echo "  ScanNet already exists at $SCANNET_DIR, skipping."
else
    echo "  ╔═══════════════════════════════════════════════════════════════╗"
    echo "  ║  ScanNet requires a Terms of Use agreement.                  ║"
    echo "  ║  Please download it manually:                                ║"
    echo "  ║    1. Follow: https://github.com/ScanNet/ScanNet             ║"
    echo "  ║    2. Download to: $SCANNET_ORIG_DIR                         ║"
    echo "  ║       python download-scannet.py -o $SCANNET_ORIG_DIR/       ║"
    echo "  ║    3. Re-run this script to convert automatically.           ║"
    echo "  ╚═══════════════════════════════════════════════════════════════╝"

    if [ -d "$SCANNET_ORIG_DIR" ] && [ "$(ls -A "$SCANNET_ORIG_DIR" 2>/dev/null)" ]; then
        echo "  Found raw ScanNet data at $SCANNET_ORIG_DIR. Converting..."
        python "$RMVD_SCRIPTS/convert_scannet.py" "$SCANNET_ORIG_DIR" "$SCANNET_DIR"
        echo "  ScanNet conversion complete."
        echo "  You can now delete $SCANNET_ORIG_DIR if desired."
    else
        echo "  Skipping ScanNet (raw data not found at $SCANNET_ORIG_DIR)."
    fi
fi

# ──────────────────────────────────────────────
# 4. Tanks and Temples
# ──────────────────────────────────────────────
echo ""
echo ">>> [4/4] Tanks and Temples"
TNT_DIR="${DATA_ROOT}/tnt"
if [ -d "$TNT_DIR" ] && [ "$(ls -A "$TNT_DIR" 2>/dev/null)" ]; then
    echo "  Tanks and Temples already exists at $TNT_DIR, skipping."
else
    echo "  Downloading Tanks and Temples..."
    bash "$RMVD_SCRIPTS/download_tanks_and_temples.sh" "$TNT_DIR"
    echo "  Tanks and Temples download complete."
fi


echo ""
echo "============================================"
echo "  Done! Data stored at: $DATA_ROOT"
echo "============================================"
echo ""
echo "Directory structure:"
echo "  data/rmvd/"
echo "  ├── eth3d/"
echo "  ├── dtu/"
echo "  ├── scannet/        (manual download required)"
echo "  └── tnt/"
