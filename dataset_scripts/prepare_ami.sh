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

    # Process each split
    for split in train dev test; do
        echo "Processing AMI $MIC_TYPE $split split..."
        
        lhotse download ami --mic "$MIC_TYPE" "$DATA_DIR/ami"
        lhotse prepare ami --mic "$MIC_TYPE" --partition full-corpus-asr --normalize-text none "$DATA_DIR/ami" "$MANIFESTS_DIR"

        manifest_prefix="ami-${MIC_TYPE}"

        # Create cutset from recordings and supervisions
        python3 "$DATA_SCRIPTS_PATH/create_cutset.py" \
            --input_recset "$MANIFESTS_DIR/${manifest_prefix}_recordings_$split.jsonl.gz" \
            --input_supset "$MANIFESTS_DIR/${manifest_prefix}_supervisions_$split.jsonl.gz" \
            --output "$MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz"
        
        # Add session prefix to IDs
        python3 "$DATA_SCRIPTS_PATH/add_prefix.py" \
            --input_manifest "$MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz" \
            --output_manifest "$MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" \
            --prefix "$MIC_TYPE"
        
        # Clean up temporary files
        rm "$MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz"
        rm "$MANIFESTS_DIR/${manifest_prefix}_supervisions_$split.jsonl.gz"
        
        # Extract supervisions from cutset
        python3 "$DATA_SCRIPTS_PATH/extract_supervisions.py" \
            --cutset_path "$MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" \
            --output_path "$MANIFESTS_DIR/${manifest_prefix}_supervisions_${split}.jsonl.gz"
    done

    python "$DATA_SCRIPTS_PATH/pre_segment_using_alignments.py" --input "$MANIFESTS_DIR/${manifest_prefix}_cutset_train.jsonl.gz" --output "$MANIFESTS_DIR/${manifest_prefix}_cutset_train_30s.jsonl.gz" --max_len 30

    echo "AMI $MIC_TYPE dataset preparation completed"
done

echo "All AMI dataset preparation completed"
