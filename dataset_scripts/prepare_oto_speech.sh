#!/bin/bash
# oto_speech ASR dataset preparation script
# Usage: prepare_oto_speech.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH

set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"

# Consistently using oto_speech!
OTO_DATA_DIR="$DATA_DIR/oto_speech"
OTO_MANIFESTS_DIR="$MANIFESTS_DIR/oto_speech"

echo "=== Preparing oto_speech dataset ==="

mkdir -p "$OTO_DATA_DIR"
mkdir -p "$OTO_MANIFESTS_DIR"

# ---------------------------------------------------------------------------
# Step 1: Download oto_speech (Audio + Parakeet v3 Pseudo-Labels)
# ---------------------------------------------------------------------------
# We check for the 'data/train' directory to see if the HuggingFace download finished
if [[ ! -d "$OTO_DATA_DIR/data/train" ]]; then
    echo "Downloading oto_speech dataset and pseudo-labels..."
    lhotse download oto-speech "$OTO_DATA_DIR"
else
    echo "oto_speech data already exists at $OTO_DATA_DIR. Skipping download."
fi

# ---------------------------------------------------------------------------
# Step 2: Prepare Lhotse Manifests
# ---------------------------------------------------------------------------
REC_FILE="$OTO_MANIFESTS_DIR/oto_recordings_train.jsonl.gz"
SUP_FILE="$OTO_MANIFESTS_DIR/oto_supervisions_train.jsonl.gz"


if [[ ! -f "$REC_FILE" ]] || [[ ! -f "$SUP_FILE" ]]; then
    echo "Preparing Lhotse manifests for oto_speech..."
    lhotse prepare oto-speech "$OTO_DATA_DIR" "$OTO_MANIFESTS_DIR"
else
    echo "Manifests already exist at $OTO_MANIFESTS_DIR. Skipping prepare."
fi

# ---------------------------------------------------------------------------
# Step 3: Create and Trim Cutsets
# ---------------------------------------------------------------------------
CUTS_FILE="$OTO_MANIFESTS_DIR/oto_cutset_train.jsonl.gz"
CUTS_PER_SEG="$OTO_MANIFESTS_DIR/oto_cutset_per_seg_train.jsonl.gz"

if [[ ! -f "$CUTS_PER_SEG" ]]; then
    echo "Creating simple cutset..."
    python "$DATA_SCRIPTS_PATH/create_cutset.py" \
        --input_recset "$REC_FILE" \
        --input_supset "$SUP_FILE" \
        --output "$CUTS_FILE"

    echo "Trimming to supervisions..."
    # Trim the full-length audio cuts down to the exact boundaries of the pseudo-labels
    lhotse cut trim-to-supervisions --discard-overlapping \
        "$CUTS_FILE" \
        "$CUTS_PER_SEG"
else
    echo "Trimmed cutset already exists. Skipping cutset generation."
fi

echo "=== Done! ==="
echo "Final trimmed cutset saved to: $CUTS_PER_SEG"
