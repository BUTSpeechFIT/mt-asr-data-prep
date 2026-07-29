#!/bin/bash

# MUSAN noise dataset preparation script
# Usage: prepare_musan.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH

set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"

MUSAN_MANIFESTS_DIR="$MANIFESTS_DIR/musan"

echo "Preparing MUSAN dataset..."

# Download MUSAN data (the corpus ships music, speech and noise together;
# we only build a cutset for the noise part, used for augmentation).
if [[ ! -d "$DATA_DIR/musan" ]]; then
    echo "Downloading MUSAN data..."
    lhotse download musan "$DATA_DIR"
fi

# Prepare MUSAN manifests
echo "Preparing MUSAN manifests..."
lhotse prepare musan "$DATA_DIR/musan" "$MUSAN_MANIFESTS_DIR"

# Build a cutset for the noise recordings so it can be loaded directly with
# lhotse's augmentation utilities (e.g. CutMix). Noise recordings have no
# supervisions.
echo "Creating cutset for MUSAN noise..."
python "$DATA_SCRIPTS_PATH/create_cutset.py" \
    --input_recset "$MUSAN_MANIFESTS_DIR/musan_recordings_noise.jsonl.gz" \
    --output "$MUSAN_MANIFESTS_DIR/musan_cutset_noise.jsonl.gz"

echo "MUSAN dataset preparation completed"
