#!/bin/bash

# LibriSpeechMix dataset preparation script
# Usage: prepare_librispeechmix.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH

set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"

LIBRISPEECHMIX_MANIFESTS_DIR="$MANIFESTS_DIR/librispeechmix"
LIBRISPEECH_MANIFESTS_DIR="$MANIFESTS_DIR/librispeech"

echo "Preparing LibriSpeechMix dataset..."

# Download LibriSpeechMix metadata if not present
if [[ ! -d "$DATA_DIR/LibriSpeechMix" ]]; then
    echo "Downloading LibriSpeechMix metadata..."
    lhotse download librispeechmix "$DATA_DIR/LibriSpeechMix"
fi

# Check if LibriSpeech manifests exist (dependency)
if [[ ! -f "$LIBRISPEECH_MANIFESTS_DIR/librispeech_recordings_train-clean-100.jsonl.gz" ]]; then
    echo "Error: LibriSpeech manifests not found. Please prepare LibriSpeech first."
    exit 1
fi

manifest_prefix="librispeechmix"
# Prepare LibriSpeechMix manifests
echo "Preparing LibriSpeechMix manifests..."
lhotse prepare librispeechmix \
    "$LIBRISPEECH_MANIFESTS_DIR" \
    "$DATA_DIR/LibriSpeechMix/list" \
    "$LIBRISPEECHMIX_MANIFESTS_DIR" \
    --num-jobs 4


echo "LibriSpeechMix dataset preparation completed."
