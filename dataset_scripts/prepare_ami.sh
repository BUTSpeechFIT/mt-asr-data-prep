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
        sdm|mdm|ihm-mix|ihm)
            ;;
        *)
            echo "Error: Invalid microphone type '$MIC_TYPE'. Supported: sdm, mdm, ihm-mix, ihm"
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

        # AMI's "ihm" recordings group every speaker's individual headset mic into one
        # multi-channel Recording (one channel per speaker). Split into one MonoCut per
        # channel -- NOT summed/downmixed -- so each cut is one speaker's clean,
        # isolated close-talk audio, matching LibriSpeech's single-speaker-per-cut shape
        # (used for pretraining).
        if [[ "$MIC_TYPE" == "ihm" ]]; then
            python "$DATA_SCRIPTS_PATH/split_multichannel_by_channel.py" \
                --input "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz" \
                --output "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz" \
                --audio_dir "$DATA_DIR/ami/ihm_split_audio/$split" \
                --num_jobs 8

            # A speaker's own headset mic still picks up faint cross-talk from other
            # speakers during overlapping speech. Trim each segment down to the parts
            # where no other speaker (per the sdm reference, which has everyone's
            # segments) was also talking, using word alignment to trim precisely.
            sdm_reference="$AMI_MANIFESTS_DIR/ami-sdm_supervisions_${split}.jsonl.gz"
            if [[ ! -f "$sdm_reference" ]]; then
                echo "Error: $sdm_reference not found. Cross-talk filtering for AMI ihm"
                echo "needs ami-sdm prepared first (bash prepare_ami.sh ... sdm)."
                exit 1
            fi
            python "$DATA_SCRIPTS_PATH/filter_crosstalk_segments.py" \
                --input "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz" \
                --reference_supset "$sdm_reference" \
                --reference_prefix_to_strip "sdm_" \
                --output "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_${split}_tmp.jsonl.gz" \
                --max_segment_duration 30
        fi

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
    if [[ "$MIC_TYPE" == "ihm" ]]; then
        # Cross-talk filtering above already emits one cut per clean utterance, trimmed
        # exactly to its span (each already <=30s) -- grouping/windowing here would
        # re-stitch cuts across the gaps where cross-talk was removed, undoing that.
        cp "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_train.jsonl.gz" \
            "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_train_30s.jsonl.gz"
    else
        echo "Preparing windowed cuts for Whisper training..."
        python "$DATA_SCRIPTS_PATH/pre_segment_using_alignments.py" --input "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_train.jsonl.gz" --output "$AMI_MANIFESTS_DIR/${manifest_prefix}_cutset_train_30s.jsonl.gz" --max_len 30 --stochastic --num_stochastic_copies 2
    fi

    echo "AMI $MIC_TYPE dataset preparation completed."
done

echo "All AMI dataset preparation completed"
