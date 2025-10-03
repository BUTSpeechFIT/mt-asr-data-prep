#!/bin/bash

# Generic AliMeeting dataset preparation script
# Usage: prepare_ali_meeting.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH [MIC_TYPES...]

set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"
shift 3
MIC_TYPES=("$@")
ALI_MEETING_MANIFESTS_DIR="$MANIFESTS_DIR/ali_meeting"

# Default to sdm if no mic types specified
if [[ ${#MIC_TYPES[@]} -eq 0 ]]; then
    MIC_TYPES=("sdm")
fi

echo "Preparing AliMeeting dataset for microphone types: ${MIC_TYPES[*]}"

# Process each microphone type
for MIC_TYPE in "${MIC_TYPES[@]}"; do
    echo "Processing AliMeeting $MIC_TYPE..."

    # Validate mic type
    case "$MIC_TYPE" in
        sdm|mdm)
            ;;
        *)
            echo "Error: Invalid microphone type '$MIC_TYPE'. Supported: sdm, mdm, ihm-mix"
            exit 1
            ;;
    esac

    if [[ ! -d "$DATA_DIR/ali_meeting" ]]; then
      lhotse download ali-meeting "$DATA_DIR/ali_meeting"
    fi
    lhotse prepare ali-meeting --mic "$MIC_TYPE" --normalize-text none "$DATA_DIR/ali_meeting" "$ALI_MEETING_MANIFESTS_DIR"

    manifest_prefix="ali_meeting-${MIC_TYPE}"

    # Process each split
    for split in train dev test; do
        echo "Processing AliMeeting $MIC_TYPE $split split..."

        # Create cutset from recordings and supervisions
        python "$DATA_SCRIPTS_PATH/create_cutset.py" \
            --input_recset "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_recordings_$split.jsonl.gz" \
            --input_supset "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_supervisions_$split.jsonl.gz" \
            --output "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz"

        # Add session prefix to IDs
        python "$DATA_SCRIPTS_PATH/add_prefix.py" \
            --input_manifest "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz" \
            --output_manifest "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" \
            --prefix "$MIC_TYPE"

        # Clean up temporary files
        rm "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz"
        rm "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_supervisions_$split.jsonl.gz"

        # Extract supervisions from cutset
        python "$DATA_SCRIPTS_PATH/extract_supervisions.py" \
            --cutset_path "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" \
            --output_path "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_supervisions_${split}.jsonl.gz"
    done

    # We cannot prepare Whisper-style data due to unavailable word alignments.

    echo "AliMeeting $MIC_TYPE dataset preparation completed."
done

echo "All AliMeeting dataset preparation completed"
