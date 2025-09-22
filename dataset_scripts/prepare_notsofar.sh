#!/bin/bash

# Generic NOTSOFAR1 dataset preparation script
# Usage: prepare_notsofar.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH [MIC_TYPES...]

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

echo "Preparing NOTSOFAR1 dataset for microphone types: ${MIC_TYPES[*]}"

# Download and prepare NOTSOFAR1 data once (if not already done)
if [[ ! -d "$DATA_DIR/notsofar" ]]; then
    echo "Downloading NOTSOFAR1 data..."
    chime-utils dgen notsofar1 "$DATA_DIR/nsf" "$DATA_DIR/notsofar" --part="train,dev,eval"
fi

# Function to process cutset for a given mic type and split
process_cutset() {
    local mic_type="$1"
    local split="$2"
    local suffix="$3"
    
    echo "Processing NOTSOFAR1 $mic_type $split split..."
    
    # Create cutset from recordings and supervisions
    python3 "$DATA_SCRIPTS_PATH/create_cutset.py" \
        --input_recset "$MANIFESTS_DIR/notsofar1-${mic_type}_recordings_${split}${suffix}.jsonl.gz" \
        --input_supset "$MANIFESTS_DIR/notsofar1-${mic_type}_supervisions_${split}${suffix}.jsonl.gz" \
        --output "$MANIFESTS_DIR/notsofar1-${mic_type}_cutset_${split}${suffix}_tmp.jsonl.gz"
    
    # Add session prefix to IDs
    python3 "$DATA_SCRIPTS_PATH/add_prefix.py" \
        --input_manifest "$MANIFESTS_DIR/notsofar1-${mic_type}_cutset_${split}${suffix}_tmp.jsonl.gz" \
        --output_manifest "$MANIFESTS_DIR/notsofar1-${mic_type}_cutset_${split}${suffix}.jsonl.gz" \
        --prefix "$mic_type"
    
    # Clean up temporary files
    rm "$MANIFESTS_DIR/notsofar1-${mic_type}_cutset_${split}${suffix}_tmp.jsonl.gz"
    rm "$MANIFESTS_DIR/notsofar1-${mic_type}_supervisions_${split}${suffix}.jsonl.gz"
    
    # Extract supervisions from cutset
    python3 "$DATA_SCRIPTS_PATH/extract_supervisions.py" \
        --cutset_path "$MANIFESTS_DIR/notsofar1-${mic_type}_cutset_${split}${suffix}.jsonl.gz" \
        --output_path "$MANIFESTS_DIR/notsofar1-${mic_type}_supervisions_${split}${suffix}.jsonl.gz"
}

# Process each microphone type
for MIC_TYPE in "${MIC_TYPES[@]}"; do
    echo "Processing NOTSOFAR1 $MIC_TYPE..."
    
    # Validate mic type and set parameters
    case "$MIC_TYPE" in
        sdm)
            DATASET_PARTS="train_sc,dev_sc,eval_sc"
            SPLITS=("train_sc" "dev_sc" "eval_sc")
            SUFFIX="_sc"
            ;;
        mdm)
            DATASET_PARTS="train,dev,eval"
            SPLITS=("train" "dev" "eval")
            SUFFIX=""
            ;;
        *)
            echo "Error: Invalid microphone type '$MIC_TYPE'. Supported: sdm, mdm"
            exit 1
            ;;
    esac

    # Prepare manifests
    chime-utils lhotse-prep notsofar1 -d "$DATASET_PARTS" --txt-norm none -m "$MIC_TYPE" "$DATA_DIR/notsofar" "$MANIFESTS_DIR"
    
    # Process cutsets for all splits
    for split in "${SPLITS[@]}"; do
        # Extract split name without suffix for processing
        split_name="${split%${SUFFIX}}"
        process_cutset "$MIC_TYPE" "$split_name" "$SUFFIX"
    done
    
    # Apply pre-segmentation using alignments for train split
    manifest_prefix="notsofar1-${MIC_TYPE}"
    train_split_suffix=""
    if [[ "$MIC_TYPE" == "sdm" ]]; then
        train_split_suffix="_sc"
    fi
    
    echo "Applying pre-segmentation to training data..."
    python3 "$DATA_SCRIPTS_PATH/pre_segment_using_alignments.py" \
        --input "$MANIFESTS_DIR/${manifest_prefix}_cutset_train${train_split_suffix}.jsonl.gz" \
        --output "$MANIFESTS_DIR/${manifest_prefix}_cutset_train${train_split_suffix}_30s.jsonl.gz" \
        --max_len 30
    
    echo "NOTSOFAR1 $MIC_TYPE dataset preparation completed"
done

echo "All NOTSOFAR1 dataset preparation completed"
