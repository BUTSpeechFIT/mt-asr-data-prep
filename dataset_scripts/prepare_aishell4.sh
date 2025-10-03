#!/bin/bash

# Generic AISHELL-4 dataset preparation script
# Usage: prepare_ali_meeting.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH [MIC_TYPES...]

set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"
shift 3
MIC_TYPES=("$@")
AISHELL4_MANIFESTS_DIR="$MANIFESTS_DIR/aishell4"

# Default to sdm if no mic types specified
if [[ ${#MIC_TYPES[@]} -eq 0 ]]; then
    MIC_TYPES=("mdm")
fi

echo "Preparing AISHELL-4 dataset for microphone types: ${MIC_TYPES[*]}"

# Process each microphone type
for MIC_TYPE in "${MIC_TYPES[@]}"; do
    echo "Processing AISHELL-4 $MIC_TYPE..."

    # Validate mic type
    case "$MIC_TYPE" in
        sdm|mdm)
            ;;
        *)
            echo "Error: Invalid microphone type '$MIC_TYPE'. Supported: sdm, mdm, ihm-mix"
            exit 1
            ;;
    esac

    if [[ ! -d "$DATA_DIR/aishell4" ]]; then
      lhotse download aishell4 "$DATA_DIR/aishell4"
    fi
    lhotse prepare aishell4 "$DATA_DIR/aishell4" "$AISHELL4_MANIFESTS_DIR"

    manifest_prefix="aishell4"

    # Process each split
    for split in train_L train_M train_S test; do
        echo "Processing AISHELL-4 $MIC_TYPE $split split..."

        # Create cutset from recordings and supervisions
        python "$DATA_SCRIPTS_PATH/create_cutset.py" \
            --input_recset "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_recordings_$split.jsonl.gz" \
            --input_supset "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_supervisions_$split.jsonl.gz" \
            --output "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz"

        # Add session prefix to IDs
        python "$DATA_SCRIPTS_PATH/add_prefix.py" \
            --input_manifest "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz" \
            --output_manifest "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" \
            --prefix "$MIC_TYPE"

        # Clean up temporary files
        rm "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz"
        rm "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_supervisions_$split.jsonl.gz"

        # Extract supervisions from cutset
        python "$DATA_SCRIPTS_PATH/extract_supervisions.py" \
            --cutset_path "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" \
            --output_path "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_supervisions_${split}.jsonl.gz"
    done

    # We cannot prepare Whisper-style data due to unavailable word alignments.

    echo "AISHELL-4 $MIC_TYPE dataset preparation completed."
done

echo "All AISHELL-4 dataset preparation completed"
