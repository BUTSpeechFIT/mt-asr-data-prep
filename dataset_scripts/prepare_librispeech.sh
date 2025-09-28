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

# Prepare LibriSpeech manifests
echo "Preparing LibriSpeech manifests..."
lhotse prepare librispeech "$DATA_DIR/librispeech/LibriSpeech" "$LIBRISPEECH_MANIFESTS_DIR"

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
