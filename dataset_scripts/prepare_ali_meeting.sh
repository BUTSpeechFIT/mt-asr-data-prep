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

    save_mono_args=()
    if [[ "$MIC_TYPE" == "sdm" ]]; then
        # Without --save-mono, lhotse's "sdm" only tags the supervision channel -- the
        # Recording itself still points at the full 8-channel far-field wav. --save-mono
        # has lhotse extract a real single-channel wav per session (via sox), so the
        # resulting cutset is genuinely mono and no downmixing is needed on our end.
        save_mono_args=(--save-mono)
    fi
    lhotse prepare ali-meeting --mic "$MIC_TYPE" --normalize-text none "${save_mono_args[@]}" "$DATA_DIR/ali_meeting" "$ALI_MEETING_MANIFESTS_DIR"

    manifest_prefix="alimeeting-${MIC_TYPE}"

    # Process each split
    for split in train test eval; do
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

    # No word-level alignments are available for AliMeeting, so unlike NOTSOFAR-1 we
    # cannot split long recordings precisely. Instead, group nearby supervisions into
    # utterance groups and drop whatever is still longer than max_len. Only "sdm" is
    # genuinely single-channel (see --save-mono above); "mdm" stays multi-channel with
    # no windowed cutset, since we don't currently train on multi-channel audio.
    if [[ "$MIC_TYPE" == "sdm" ]]; then
        echo "Preparing windowed cuts for Whisper training..."
        python "$DATA_SCRIPTS_PATH/trim_to_supervision_groups.py" \
            --input "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_train.jsonl.gz" \
            --output "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_train_grouped.jsonl.gz" \
            --max_pause 2 --stochastic --num_stochastic_copies 2 --offset_window 30

        python "$DATA_SCRIPTS_PATH/filter_by_length.py" \
            --input "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_train_grouped.jsonl.gz" \
            --output "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_train_30s.jsonl.gz" \
            --max_len 30

        rm "$ALI_MEETING_MANIFESTS_DIR/${manifest_prefix}_cutset_train_grouped.jsonl.gz"
    fi

    echo "AliMeeting $MIC_TYPE dataset preparation completed."
done

echo "All AliMeeting dataset preparation completed"
