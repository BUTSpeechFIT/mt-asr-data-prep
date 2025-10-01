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

NOTSOFAR_MANIFESTS_DIR="${MANIFESTS_DIR}/notsofar1"
VERSIONS=("240825.1_train" "240825.1_dev1" "240629.1_eval_small_with_GT")
SPLITS=("train_set_${VERSIONS[0]}" "dev_set_${VERSIONS[1]}" "eval_set_${VERSIONS[2]}")

# Default to sdm if no mic types specified
if [[ ${#MIC_TYPES[@]} -eq 0 ]]; then
    MIC_TYPES=("sdm")
fi

echo "Preparing NOTSOFAR1 dataset for microphone types: ${MIC_TYPES[*]}"

# Download and prepare NOTSOFAR1 data once (if not already done)
# if [[ ! -d "$DATA_DIR/nsf" ]]; then - this IF does not make sense if we're downloading to the same directory.
echo "Downloading NOTSOFAR1 data (if not done already)..."
    # chime-utils dgen notsofar1 "$DATA_DIR/nsf" "$DATA_DIR/notsofar" --part="train,dev,eval" --download --txt-norm none
lhotse download notsofar1 -p train -p dev -p test --mic $MIC_TYPES --train-version "${VERSIONS[0]}" --dev-version "${VERSIONS[1]}" --test-version "${VERSIONS[2]}" "$DATA_DIR/nsf"
# fi

# Function to process cutset for a given mic type and split
process_cutset() {
    local mic_type="$1"
    local split="$2"

    echo "Processing NOTSOFAR1 $mic_type $split split..."

    # Create cutset from recordings and supervisions
    python "$DATA_SCRIPTS_PATH/create_cutset.py" \
        --input_recset "$NOTSOFAR_MANIFESTS_DIR/notsofar1_${mic_type}_${split}_recordings.jsonl.gz" \
        --input_supset "$NOTSOFAR_MANIFESTS_DIR/notsofar1_${mic_type}_${split}_supervisions.jsonl.gz" \
        --output "$NOTSOFAR_MANIFESTS_DIR/notsofar1_${mic_type}_${split}_cutset_tmp.jsonl.gz"

    # Add session prefix to IDs
    python "$DATA_SCRIPTS_PATH/add_prefix.py" \
        --input_manifest "$NOTSOFAR_MANIFESTS_DIR/notsofar1_${mic_type}_${split}_cutset_tmp.jsonl.gz" \
        --output_manifest "$NOTSOFAR_MANIFESTS_DIR/notsofar1_${mic_type}_${split}_cutset.jsonl.gz" \
        --prefix "$mic_type"

    # Clean up temporary files
    rm "$NOTSOFAR_MANIFESTS_DIR/notsofar1_${mic_type}_${split}_cutset_tmp.jsonl.gz"
    rm "$NOTSOFAR_MANIFESTS_DIR/notsofar1_${mic_type}_${split}_supervisions.jsonl.gz"

    # Extract supervisions from cutset
    python "$DATA_SCRIPTS_PATH/extract_supervisions.py" \
        --cutset_path "$NOTSOFAR_MANIFESTS_DIR/notsofar1_${mic_type}_${split}_cutset.jsonl.gz" \
        --output_path "$NOTSOFAR_MANIFESTS_DIR/notsofar1_${mic_type}_${split}_supervisions.jsonl.gz"
}

# Process each microphone type
for MIC_TYPE in "${MIC_TYPES[@]}"; do
    echo "Processing NOTSOFAR1 $MIC_TYPE..."

    # Prepare manifests
    lhotse prepare notsofar1 "$DATA_DIR/nsf" "$NOTSOFAR_MANIFESTS_DIR"

    # Process cutsets for all splits
    for split in "${SPLITS[@]}"; do
        # Extract split name without suffix for processing
        process_cutset "$MIC_TYPE" "$split"
    done

#    # Apply pre-segmentation using alignments for train split
    manifest_prefix="notsofar1_${MIC_TYPE}"

    echo "$NOTSOFAR_MANIFESTS_DIR/${manifest_prefix}_${SPLITS[0]}_cutset.jsonl.gz"

    echo "Preparing windowed cuts for Whisper training..."
    python "$DATA_SCRIPTS_PATH/pre_segment_using_alignments.py" \
        --input "$NOTSOFAR_MANIFESTS_DIR/${manifest_prefix}_${SPLITS[0]}_cutset.jsonl.gz" \
        --output "$NOTSOFAR_MANIFESTS_DIR/${manifest_prefix}_${SPLITS[0]}_cutset_30s.jsonl.gz" \
        --max_len 30

    echo "NOTSOFAR1 $MIC_TYPE dataset preparation completed"
done

echo "All NOTSOFAR1 dataset preparation completed"
