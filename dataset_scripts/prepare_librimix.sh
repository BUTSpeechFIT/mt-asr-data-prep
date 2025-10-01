#!/bin/bash

# LibriMix dataset preparation script
# Usage: prepare_librimix.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH

set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"

LIBRIMIX_MANIFESTS_DIR="$MANIFESTS_DIR/librimix"
LIBRISPEECH_MANIFESTS_DIR="$MANIFESTS_DIR/librispeech"
WHAM_MANIFESTS_DIR="$MANIFESTS_DIR/wham_noise"

echo "Preparing LibriMix dataset..."

# Download WHAM noise if not present
if [[ ! -d "$DATA_DIR/wham_noise" ]]; then
    echo "Downloading WHAM noise..."
    lhotse download wham "$DATA_DIR/wham_noise"
fi

# Prepare WHAM noise manifests if not present
if [[ ! -f "$WHAM_MANIFESTS_DIR/wham_recordings_tr.jsonl.gz" ]]; then
    echo "Preparing WHAM noise manifests..."
    lhotse prepare wham "$DATA_DIR/wham_noise/wham_noise" "$WHAM_MANIFESTS_DIR"
fi

# Download LibriMix metadata if not present
if [[ ! -d "$DATA_DIR/LibriMix/metadata" ]]; then
    echo "Downloading LibriMix metadata..."
    lhotse download librimix "$DATA_DIR/LibriMix"
fi

# Check if LibriSpeech manifests exist (dependency)
if [[ ! -f "$LIBRISPEECH_MANIFESTS_DIR/librispeech_recordings_train-clean-100.jsonl.gz" ]]; then
    echo "Error: LibriSpeech manifests not found. Please prepare LibriSpeech first."
    exit 1
fi

manifest_prefix="librimix"
# Prepare LibriMix manifests
echo "Preparing LibriMix manifests..."
for n_src in 2 3; do
  lhotse prepare librimix \
      "$LIBRISPEECH_MANIFESTS_DIR" \
      "$WHAM_MANIFESTS_DIR" \
      "$DATA_DIR/LibriMix/metadata" \
      "$DATA_DIR/LibriMix/storage" \
      "$LIBRIMIX_MANIFESTS_DIR" \
      --n_src $n_src \
      --num-jobs 4

  for split in clean-100 clean-360 clean-100_noisy clean-360_noisy; do
    if [[ ! -f "$LIBRIMIX_MANIFESTS_DIR/${manifest_prefix}_cutset_libri${n_src}mix_train-${split}_30s.jsonl.gz" ]]; then
      echo "Preparing 30s windowed cuts for split libri${n_src}mix_train-${split} for Whisper training..."
      python "$DATA_SCRIPTS_PATH/filter_by_length.py" \
        --input "$LIBRIMIX_MANIFESTS_DIR/${manifest_prefix}_cutset_libri${n_src}mix_train-${split}.jsonl.gz" \
        --output "$LIBRIMIX_MANIFESTS_DIR/${manifest_prefix}_cutset_libri${n_src}mix_train-${split}_30s.jsonl.gz" \
        --max_len 30
    fi
  done
done

echo "LibriMix dataset preparation completed."
