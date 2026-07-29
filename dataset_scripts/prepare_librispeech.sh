#!/bin/bash

# LibriSpeech dataset preparation script
# Usage: prepare_librispeech.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH

set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"

LIBRISPEECH_MANIFESTS_DIR="$MANIFESTS_DIR/librispeech"

echo "Preparing LibriSpeech dataset..."

# Download and prepare LibriSpeech data
if [[ ! -d "$DATA_DIR/librispeech" ]]; then
    echo "Downloading LibriSpeech data..."
    lhotse download librispeech "$DATA_DIR/librispeech"
fi

# Download word-level alignments (CorentinJ/librispeech-alignments) and merge them into
# the LibriSpeech directory. The "lhotse download librispeech" CLI has no --alignments
# flag, so this goes through the Python API instead. Needed by tools that place speaker
# turns precisely from word timings (e.g. FastMSS).
# The upstream Google Drive link is frequently rate-limited ("too many accesses"), so this
# defaults to a personal mirror; override via LIBRISPEECH_ALIGNMENTS_URL if it goes stale.
LIBRISPEECH_ALIGNMENTS_URL="${LIBRISPEECH_ALIGNMENTS_URL:-https://drive.google.com/uc?id=10uii01U3XqIw2_x2H9h5t7-wUb2JnGn4}"
if [[ ! -f "$DATA_DIR/librispeech/.ali_completed" ]]; then
    echo "Downloading LibriSpeech word alignments..."
    python "$DATA_SCRIPTS_PATH/download_librispeech_alignments.py" \
        --target_dir "$DATA_DIR/librispeech" \
        --alignments_url "$LIBRISPEECH_ALIGNMENTS_URL"
fi

# Prepare LibriSpeech manifests
# Note: lhotse skips parts whose manifests already exist in $LIBRISPEECH_MANIFESTS_DIR.
# If you ran this script before this alignments support was added, delete
# $LIBRISPEECH_MANIFESTS_DIR and re-run to regenerate supervisions with alignments.
echo "Preparing LibriSpeech manifests..."
lhotse prepare librispeech --alignments-dir "$DATA_DIR/librispeech/LibriSpeech" "$DATA_DIR/librispeech/LibriSpeech" "$LIBRISPEECH_MANIFESTS_DIR"

manifest_prefix="librispeech"

# Process each split
for split in train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other; do
    if [[ ! -f "$LIBRISPEECH_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" ]]; then
      echo "Creating cutset for LibriSpeech $split split..."
      # Create cutset from recordings and supervisions
      python "$DATA_SCRIPTS_PATH/create_cutset.py" \
          --input_recset "$LIBRISPEECH_MANIFESTS_DIR/${manifest_prefix}_recordings_$split.jsonl.gz" \
          --input_supset "$LIBRISPEECH_MANIFESTS_DIR/${manifest_prefix}_supervisions_$split.jsonl.gz" \
          --output "$LIBRISPEECH_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz"
    fi
done

echo "LibriSpeech dataset preparation completed"
