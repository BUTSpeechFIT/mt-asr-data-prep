#!/bin/bash
# VoxPopuli ASR dataset preparation script with MFA Forced Alignment
# Usage: prepare_voxpopuli.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH GDRIVE_FILE_ID
#set -euo pipefail

# Arguments
DATA_DIR="$1"
MANIFESTS_DIR="$2"
DATA_SCRIPTS_PATH="$3"
GDRIVE_FILE_ID=https://drive.google.com/file/d/1JQ2TxjPgodfYYCgkwtDtlcTzl9yRMarN/view?usp=sharing

shift 4 || true

SUBSET="asr"
LANG="en"
VOXPOPULI_MANIFESTS_DIR="$MANIFESTS_DIR/voxpopuli"

echo "Preparing VoxPopuli dataset (Subset: $SUBSET) for language: $LANG"

mkdir -p "$DATA_DIR/voxpopuli"
mkdir -p "$VOXPOPULI_MANIFESTS_DIR"

# ---------------------------------------------------------------------------
# Step 1: Download and Extract MFA Alignments from Google Drive
# ---------------------------------------------------------------------------
MFA_TAR_FILE="$DATA_DIR/voxpopuli/voxpopuli_mfa_alignments.tar.gz"
MFA_EXTRACT_DIR="$DATA_DIR/voxpopuli/mfa_alignments"

if [[ ! -d "$MFA_EXTRACT_DIR" ]]; then
    echo "Downloading MFA alignments from Google Drive..."
    gdown --fuzzy "$GDRIVE_FILE_ID" -O "$MFA_TAR_FILE"

    echo "Extracting alignments..."
    mkdir -p "$MFA_EXTRACT_DIR"
    tar -xzvf "$MFA_TAR_FILE" -C "$DATA_DIR/voxpopuli"
else
    echo "MFA Alignments already exist at $MFA_EXTRACT_DIR. Skipping download."
fi

# Download specifically the ASR subset
if [[ ! -d "$DATA_DIR/voxpopuli/raw_audios" ]]; then
    echo "Downloading VoxPopuli $LANG ($SUBSET)..."
    lhotse download voxpopuli "$DATA_DIR/voxpopuli" --subset "$SUBSET"
fi

if [[ ! -d "$DATA_DIR/voxpopuli/raw_audios/${LANG}" ]]; then
    echo "Due to bug in prep code, moving from original to ${LANG} subdir"
    mv "$DATA_DIR/voxpopuli/raw_audios/original" "$DATA_DIR/voxpopuli/raw_audios/${LANG}"
fi

# ---------------------------------------------------------------------------
# Step 2: Prepare Lhotse Manifests
# ---------------------------------------------------------------------------
echo "Preparing Lhotse manifests for $LANG..."
lhotse prepare voxpopuli "$DATA_DIR/voxpopuli" "$VOXPOPULI_MANIFESTS_DIR" --task "$SUBSET" --lang "$LANG"

manifest_prefix="voxpopuli"
for split in train dev test; do
    echo "Processing VoxPopuli $LANG $split split..."

    REC_FILE="$VOXPOPULI_MANIFESTS_DIR/${manifest_prefix}-asr-${LANG}_recordings_${split}.jsonl.gz"
    SUP_FILE="$VOXPOPULI_MANIFESTS_DIR/${manifest_prefix}-asr-${LANG}_supervisions_${split}.jsonl.gz"
    # New file for the swapped text
    SUP_FILE_ORIG="$VOXPOPULI_MANIFESTS_DIR/${manifest_prefix}-asr-${LANG}_supervisions_orig_text_${split}.jsonl.gz"

    if [[ ! -f "$REC_FILE" ]]; then
        echo "Warning: $REC_FILE not found. Skipping $split split."
        continue
    fi

    # --- Step 3a: Post-process to use orig_text instead of normed_text ---
    echo "  Replacing normed_text with orig_text for $split..."
    python3 - <<EOF
from lhotse import SupervisionSet

def use_orig_text(sup):
  orig = sup.custom.pop("orig_text")
  if orig:
    sup.text = orig
    sup.custom = None
  return sup

sups = SupervisionSet.from_file("$SUP_FILE")
sups = sups.map(use_orig_text)
sups.to_file("$SUP_FILE_ORIG")
EOF

    CUTS_FILE="$VOXPOPULI_MANIFESTS_DIR/${manifest_prefix}-asr-${LANG}_cuts_${split}.jsonl.gz"
    CUTS_PER_SEG="$VOXPOPULI_MANIFESTS_DIR/${manifest_prefix}-asr-${LANG}_cuts_per_segment_${split}.jsonl.gz"
    CUTS_ALIGNED="$VOXPOPULI_MANIFESTS_DIR/${manifest_prefix}-asr-${LANG}_cuts_per_segment_aligned_${split}.jsonl.gz"

    # --- Step 3b: build simple cuts (using the new orig_text supervisions) ---
    python "$DATA_SCRIPTS_PATH/create_cutset.py" \
        --input_recset "$REC_FILE" \
        --input_supset "$SUP_FILE_ORIG" \
        --output  "$CUTS_FILE"

    # --- Step 3c: trim to supervisions ---
    lhotse cut trim-to-supervisions --discard-overlapping \
        "$CUTS_FILE" \
        "$CUTS_PER_SEG"

    # --- Step 3d: merge TextGrids into Lhotse cuts natively ---
    echo "  [FA] Merging MFA TextGrids into Lhotse cuts natively for $split..."

    python "$DATA_SCRIPTS_PATH/merge_mfa_into_cuts.py" \
        --cuts_file "$CUTS_PER_SEG" \
        --textgrid_dir "$MFA_EXTRACT_DIR/$split" \
        --out_cuts "$CUTS_ALIGNED"

    echo "  Done: $CUTS_ALIGNED"
done

echo "Finished"
