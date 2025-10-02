#!/bin/bash

# Generic AMI dataset preparation script
# Usage: prepare_ami.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH [MIC_TYPES...]

set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"
shift 3
MIC_TYPES=("$@")
AMI_MANIFESTS_DIR="$MANIFESTS_DIR/ami"

# Default to sdm if no mic types specified
if [[ ${#MIC_TYPES[@]} -eq 0 ]]; then
    MIC_TYPES=("sdm")
fi

echo "Preparing AMI dataset for microphone types: ${MIC_TYPES[*]}"

# Process each microphone type
for MIC_TYPE in "${MIC_TYPES[@]}"; do
    echo "Processing AMI $MIC_TYPE..."

    # Validate mic type
    case "$MIC_TYPE" in
        sdm|mdm|ihm-mix)
            ;;
        *)
            echo "Error: Invalid microphone type '$MIC_TYPE'. Supported: sdm, mdm, ihm-mix"
            exit 1
            ;;
    esac

    if [[ ! -d "$DATA_DIR/ami/${MIC_TYPE}" ]]; then
      lhotse download ami --mic "$MIC_TYPE" "$DATA_DIR/ami/${MIC_TYPE}"
    fi
    lhotse prepare ami --mic "$MIC_TYPE" --partition full-corpus-asr --normalize-text none --keep-punctuation "$DATA_DIR/ami/${MIC_TYPE}" "$AMI_MANIFESTS_DIR"

    manifest_prefix="ami-${MIC_TYPE}"

    # Process each split
    for split in train dev test; do
        echo "Processing AMI $MIC_TYPE $split split..."

        # Create cutset from recordings and supervisions
        python "$DATA_SCRIPTS_PATH/create_cutset.py" \
            --input_recset "$AMI_MANIFESTS_DIR/${manifest_prefix}_recordings_$split.jsonl.gz" \
            --input_supset "$AMI_MANIFESTS_DIR/${manifest_prefix}_supervisions_$split.jsonl.gz" \
            --output "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz"

        # Add session prefix to IDs
        python "$DATA_SCRIPTS_PATH/add_prefix.py" \
            --input_manifest "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz" \
            --output_manifest "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" \
            --prefix "$MIC_TYPE"

        # Clean up temporary files
        rm "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz"
        rm "$AMI_MANIFESTS_DIR/${manifest_prefix}_supervisions_$split.jsonl.gz"

        # Extract supervisions from cutset
        python "$DATA_SCRIPTS_PATH/extract_supervisions.py" \
            --cutset_path "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" \
            --output_path "$AMI_MANIFESTS_DIR/${manifest_prefix}_supervisions_${split}.jsonl.gz"
    done

    # Prepare windowed cuts for Whisper training
    echo "Preparing windowed cuts for Whisper training..."

    python "$DATA_SCRIPTS_PATH/pre_segment_using_alignments.py" --input "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_train.jsonl.gz" --output "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_train_30s.jsonl.gz" --max_len 30

    echo "AMI $MIC_TYPE dataset preparation completed."
done

echo "All AMI dataset preparation completed"
