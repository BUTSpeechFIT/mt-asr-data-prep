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


if [[ ! -f "$LIBRISPEECHMIX_MANIFESTS_DIR/librispeechmix_custom_cutset_train-3mix.jsonl.gz" ]]; then
  
  # Download prepared librispeechmix-like training set
  echo "Downloading prepared LibriSpeechMix-like training set."
  curl -L -o "$LIBRISPEECHMIX_MANIFESTS_DIR/librispeechmix3_download.jsonl.gz" https://nextcloud.fit.vutbr.cz/public.php/dav/files/ddRxWNZJ4Aw6gwy/?accept=zip
  python "$DATA_SCRIPTS_PATH/change_sources_prefix.py" --input_manifest "$LIBRISPEECHMIX_MANIFESTS_DIR/librispeechmix3_download.jsonl.gz" --output_manifest "$LIBRISPEECHMIX_MANIFESTS_DIR/librispeechmix_custom_cutset_train-3mix.jsonl.gz" --orig_prefix PREFIX/ --new_prefix $DATA_DIR/
  
  # Alternatively create custom simulated mixtures
  #python "$DATA_SCRIPTS_PATH/create_synthetic_ls_mixtures.py" --input_manifests "$LIBRISPEECH_MANIFESTS_DIR/librispeech_cutset_train-clean-100.jsonl.gz"  "$LIBRISPEECH_MANIFESTS_DIR/librispeech_cutset_train-clean-360.jsonl.gz" "$LIBRISPEECH_MANIFESTS_DIR/librispeech_cutset_train-other-500.jsonl.gz"  --output_manifest "$LIBRISPEECHMIX_MANIFESTS_DIR/librispeechmix_custom_cutset_train-3mix_.jsonl.gz"
fi 


echo "LibriSpeechMix dataset preparation completed."
