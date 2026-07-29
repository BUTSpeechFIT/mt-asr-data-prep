#!/bin/bash

# FastMSS synthetic multi-speaker meeting simulation script
# Synthesizes conversational mixtures (up to 5 speakers, 30s windows by default) from
# LibriSpeech using https://github.com/popcornell/FastMSS
# Usage: prepare_fastmss.sh DATA_DIR MANIFESTS_DIR DATA_SCRIPTS_PATH
#
# Tunable via environment variables:
#   FASTMSS_N_MEETINGS   number of meetings to generate (default: 1000)
#   FASTMSS_DURATION     target meeting duration in seconds (default: 30)
#   FASTMSS_MIN_SPK      minimum speakers per meeting (default: 2; FastMSS's turn-taking
#                        model always tries to switch to "a different speaker", which
#                        requires at least 2 speakers in the sampled pool -- 1 crashes)
#   FASTMSS_MAX_SPK      maximum speakers per meeting (default: 5)
#   FASTMSS_N_JOBS       parallel workers (default: 8)

set -euo pipefail

# Arguments
# Resolved to absolute paths: recipes/sim.py uses Hydra with hydra.run.dir=${output_dir},
# which chdirs the process into output_dir before the job runs. A relative manifest_dir
# (resolved against the original launch directory) would then point at the wrong place.
DATA_DIR="$(cd "$1" && pwd)"
MANIFESTS_DIR="$(cd "$2" && pwd)"
DATA_SCRIPTS_PATH="$(cd "$3" && pwd)"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTMSS_DIR="$REPO_ROOT/FastMSS"
LIBRISPEECH_MANIFESTS_DIR="$MANIFESTS_DIR/librispeech"
FASTMSS_MANIFESTS_DIR="$MANIFESTS_DIR/fastmss"
FASTMSS_OUTPUT_DIR="$DATA_DIR/fastmss"

N_MEETINGS="${FASTMSS_N_MEETINGS:-1000}"
DURATION="${FASTMSS_DURATION:-30}"
MIN_SPK="${FASTMSS_MIN_SPK:-2}"
MAX_SPK="${FASTMSS_MAX_SPK:-5}"
N_JOBS="${FASTMSS_N_JOBS:-8}"

echo "Preparing FastMSS synthetic meetings ($N_MEETINGS meetings, ${DURATION}s windows, ${MIN_SPK}-${MAX_SPK} speakers)..."

# Clone FastMSS if not already present
if [[ ! -d "$FASTMSS_DIR" ]]; then
    echo "Cloning FastMSS..."
    git clone https://github.com/popcornell/FastMSS.git "$FASTMSS_DIR"
fi

# Install FastMSS (idempotent; pip is a no-op if already satisfied)
echo "Installing FastMSS..."
pip install -e "$FASTMSS_DIR" -q

# Check dependency: LibriSpeech manifests with word-level alignments (see
# prepare_librispeech.sh; FastMSS needs word timings to place speaker turns/overlaps).
if [[ ! -f "$LIBRISPEECH_MANIFESTS_DIR/librispeech_cutset_train-clean-100.jsonl.gz" ]]; then
    echo "Error: LibriSpeech manifests not found in $LIBRISPEECH_MANIFESTS_DIR."
    echo "Please prepare LibriSpeech first (dataset_scripts/prepare_librispeech.sh)."
    exit 1
fi

mkdir -p "$FASTMSS_OUTPUT_DIR" "$FASTMSS_MANIFESTS_DIR"

# recipes/sim.py uses Hydra with config_path="." (relative to the file itself), and is
# documented to be invoked from within the FastMSS repo root.
(
    cd "$FASTMSS_DIR"
    python recipes/sim.py \
        output_dir="$FASTMSS_OUTPUT_DIR" \
        manifest_dir="$LIBRISPEECH_MANIFESTS_DIR" \
        n_meetings="$N_MEETINGS" \
        duration="$DURATION" \
        min_max_spk="[$MIN_SPK,$MAX_SPK]" \
        n_jobs="$N_JOBS"
)

# Expose the simulated manifests alongside the rest of this repo's prepared datasets.
manifest_prefix="fastmss"
cp "$FASTMSS_OUTPUT_DIR/manifests/synth-librispeech-train-recordings.jsonl.gz" \
    "$FASTMSS_MANIFESTS_DIR/${manifest_prefix}_recordings_train.jsonl.gz"
cp "$FASTMSS_OUTPUT_DIR/manifests/synth-librispeech-train-supervisions.jsonl.gz" \
    "$FASTMSS_MANIFESTS_DIR/${manifest_prefix}_supervisions_train.jsonl.gz"
cp "$FASTMSS_OUTPUT_DIR/manifests/synth-librispeech-train-cuts.jsonl.gz" \
    "$FASTMSS_MANIFESTS_DIR/${manifest_prefix}_cutset_train.jsonl.gz"

# Meetings are windowed to a *target* duration -- FastMSS extends the last turn to
# finish rather than cutting it off, so actual cut length is always >= DURATION (and can
# run well past it once overlapping turns are involved). Window down to <=30s using the
# LibriSpeech word alignments carried over onto each supervision, same as AMI/NOTSOFAR-1.
echo "Preparing windowed cuts for Whisper training..."
python "$DATA_SCRIPTS_PATH/pre_segment_using_alignments.py" \
    --input "$FASTMSS_MANIFESTS_DIR/${manifest_prefix}_cutset_train.jsonl.gz" \
    --output "$FASTMSS_MANIFESTS_DIR/${manifest_prefix}_cutset_train_30s.jsonl.gz" \
    --max_len 30 --stochastic --num_stochastic_copies 2

echo "FastMSS dataset preparation completed."
