#!/bin/bash

# Exit on any error
set -e

# The 6 domains from the paper
DOMAINS=("PER" "HSN" "SSW" "SNE" "NES" "UHH")
DATA_DIR="./data"

# Check if we are in the right directory
if [ ! -d "preprocessing" ]; then
    echo "Error: Please run this script from the BIRDSET_ALL_DOMAIN parent folder."
    exit 1
fi

echo "================================================="
echo "   SG-MSM Data Preprocessing Automation Script"
echo "================================================="
echo "Domains to process: ${DOMAINS[*]}"
echo ""

for DOMAIN in "${DOMAINS[@]}"; do
    echo "-------------------------------------------------"
    echo "🚀 Starting full preprocessing for domain: $DOMAIN"
    echo "-------------------------------------------------"

    echo "[1/3] Downloading Data for $DOMAIN..."
    python3 preprocessing/download_data.py --domain "$DOMAIN" --data_dir "$DATA_DIR"

    echo "[2/3] Mining Background Noise for $DOMAIN..."
    python3 preprocessing/noise_mining.py --domain "$DOMAIN" --data_dir "$DATA_DIR"

    echo "[3/3] Converting Audio to NPY for $DOMAIN..."
    python3 preprocessing/convert_to_npy.py --domain "$DOMAIN" --data_dir "$DATA_DIR"

    echo "✅ Successfully preprocessed domain: $DOMAIN"
    echo ""
done

echo "🎉 All 6 domains have been completely preprocessed!"
echo "You can now run training for any domain."
