#!/bin/bash
# SkyGuard Dataset Download Script
# Downloads drone audio datasets from public sources

set -e  # Exit on error

echo "============================================================"
echo "SkyGuard Data Acquisition"
echo "============================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$SCRIPT_DIR/../.."
RAW_DIR="$BASE_DIR/data/raw"

# Create data directory
mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

echo "Data directory: $RAW_DIR"
echo ""

# 1. Download DroneAudioset (Primary dataset - 23.5 hours)
echo "============================================================"
echo "1. Downloading DroneAudioset (23.5 hours, MIT license)"
echo "============================================================"
if [ ! -d "DroneAudioSet" ]; then
    echo "Cloning from HuggingFace..."
    git clone https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet DroneAudioSet
    echo "✓ DroneAudioset cloned"
else
    echo "✓ DroneAudioset already exists"
fi
echo ""

# 2. Download Multiclass Drone Dataset (Backup - 3,200 recordings)
echo "============================================================"
echo "2. Downloading Multiclass Drone Dataset (3,200 recordings)"
echo "============================================================"
if [ ! -d "DroneAudioDataset" ]; then
    echo "Cloning from GitHub..."
    git clone https://github.com/saraalemadi/DroneAudioDataset.git DroneAudioDataset
    echo "✓ Multiclass dataset cloned"
else
    echo "✓ Multiclass dataset already exists"
fi
echo ""

# 3. Download DroneAudioSet code repository
echo "============================================================"
echo "3. Downloading DroneAudioSet tools/code"
echo "============================================================"
if [ ! -d "DroneAudioSet-code" ]; then
    echo "Cloning tools repository..."
    git clone https://github.com/augmented-human-lab/DroneAudioSet-code.git DroneAudioSet-code
    echo "✓ Tools repository cloned"
else
    echo "✓ Tools repository already exists"
fi
echo ""

# 4. Create dataset structure
echo "============================================================"
echo "4. Creating training dataset structure"
echo "============================================================"

for split in train val test; do
    mkdir -p "$split/Non-Drone"
    for i in {1..10}; do
        mkdir -p "$split/Drone_Model_$i"
    done
done

echo "✓ Dataset structure created:"
echo "  - data/raw/train/ (11 classes)"
echo "  - data/raw/val/ (11 classes)"
echo "  - data/raw/test/ (11 classes)"
echo ""

# 5. Download ESC-50 for non-drone sounds
echo "============================================================"
echo "5. Downloading ESC-50 (environmental sounds for Non-Drone class)"
echo "============================================================"
if [ ! -d "ESC-50" ]; then
    echo "Downloading ESC-50 dataset..."
    curl -L "https://github.com/karolpiczak/ESC-50/archive/master.zip" -o esc50.zip
    unzip -q esc50.zip
    mv ESC-50-master ESC-50
    rm esc50.zip
    echo "✓ ESC-50 downloaded (2,000 environmental sounds)"
else
    echo "✓ ESC-50 already exists"
fi
echo ""

# Summary
echo "============================================================"
echo "✓ ALL DATASETS DOWNLOADED SUCCESSFULLY!"
echo "============================================================"
echo ""
echo "Downloaded datasets:"
echo "  1. DroneAudioSet - 23.5 hours of drone audio"
echo "  2. Multiclass Dataset - 3,200 drone recordings"
echo "  3. ESC-50 - 2,000 environmental sounds"
echo ""
echo "Location: $RAW_DIR"
echo ""
echo "Next steps:"
echo "  1. Review datasets in data/raw/"
echo "  2. Run: python3 backend/data_prep/organize_for_training.py"
echo "  3. Train model: python3 backend/training/train.py"
echo ""
echo "============================================================"
