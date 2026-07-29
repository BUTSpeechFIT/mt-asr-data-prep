#!/bin/bash

# Generic AISHELL-4 dataset preparation script
# Usage: prepare_aishell4.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH [MIC_TYPES...]
#
# Tunable via environment variables:
#   AISHELL4_DOWNMIX   whether to build a downmixed single-channel version on top of the
#                       raw 8-channel array cutset (default: true). AISHELL-4 has no
#                       native single-channel release, so this is the only way to get
#                       single-channel training data out of it; disable it if you only
#                       want the raw multi-channel cutset.

set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"
shift 3
MIC_TYPES=("$@")
AISHELL4_MANIFESTS_DIR="$MANIFESTS_DIR/aishell4"
AISHELL4_DOWNMIX="${AISHELL4_DOWNMIX:-true}"

# AISHELL-4 only ships the raw 8-channel array recordings -- no native single-channel
# release exists (unlike AliMeeting's "sdm"), so "mdm" is the only real mic type.
if [[ ${#MIC_TYPES[@]} -eq 0 ]]; then
    MIC_TYPES=("mdm")
fi

echo "Preparing AISHELL-4 dataset for microphone types: ${MIC_TYPES[*]}"

# Process each microphone type
for MIC_TYPE in "${MIC_TYPES[@]}"; do
    echo "Processing AISHELL-4 $MIC_TYPE..."

    # Validate mic type
    case "$MIC_TYPE" in
        mdm)
            ;;
        *)
            echo "Error: Invalid microphone type '$MIC_TYPE'. Supported: mdm"
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

    # AISHELL-4 recordings are always 8-channel. On top of the original multi-channel
    # cutset, build a downmixed (summed-to-mono) version and cache it to a single audio
    # file per session, so future loading doesn't need to read+mix all 8 channels.
    if [[ "$AISHELL4_DOWNMIX" == "true" ]]; then
        echo "Creating downmixed single-channel version of AISHELL-4 $MIC_TYPE..."

        for split in train_L train_M train_S test; do
            echo "Downmixing AISHELL-4 $MIC_TYPE $split split to single channel..."

            python "$DATA_SCRIPTS_PATH/downmix_recordings.py" \
                --input "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}.jsonl.gz" \
                --output "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_downmix.jsonl.gz" \
                --audio_dir "$DATA_DIR/aishell4/downmix_audio/$split" \
                --num_jobs 8

            python "$DATA_SCRIPTS_PATH/extract_supervisions.py" \
                --cutset_path "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_downmix.jsonl.gz" \
                --output_path "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_supervisions_${split}_downmix.jsonl.gz"
        done

        # No word-level alignments are available for AISHELL-4, so unlike NOTSOFAR-1 we
        # cannot split long recordings precisely. Instead, group nearby supervisions into
        # utterance groups and drop whatever is still longer than max_len.
        echo "Preparing windowed cuts for Whisper training..."
        for split in train_L train_M train_S; do
            python "$DATA_SCRIPTS_PATH/trim_to_supervision_groups.py" \
                --input "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_downmix.jsonl.gz" \
                --output "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_downmix_grouped.jsonl.gz" \
                --max_pause 2 --stochastic --num_stochastic_copies 2 --offset_window 30

            python "$DATA_SCRIPTS_PATH/filter_by_length.py" \
                --input "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_downmix_grouped.jsonl.gz" \
                --output "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_downmix_30s.jsonl.gz" \
                --max_len 30

            rm "$AISHELL4_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_downmix_grouped.jsonl.gz"
        done
    fi

    echo "AISHELL-4 $MIC_TYPE dataset preparation completed."
done

echo "All AISHELL-4 dataset preparation completed"
