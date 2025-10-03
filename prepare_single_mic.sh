#!/bin/bash

# Single-microphone dataset preparation script
# Usage: ./prepare_single_mic.sh [OPTIONS]

set -euo pipefail

# Constants
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
readonly DATASET_SCRIPTS_DIR="$SCRIPT_DIR/dataset_scripts"

readonly AVAILABLE_DATASETS=(
    "librispeech" "librimix" "librispeechmix" "ali_meeting-sdm" "ami-sdm" "ami-ihm-mix" "notsofar1-sdm"
)

# Dataset dependencies (bash 3 compatible)
get_dataset_dependency() {
    case "$1" in
        librimix)
            echo "librispeech"
            ;;
        librispeechmix)
            echo "librispeech"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Default configuration
DATASETS="all"
EXTRACT_SUPERVISIONS=false
ROOT_DIR="$(pwd)"
VERBOSE=false

# ...existing code... (logging functions, usage, parse_arguments)

# Usage function
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Options:
  -d, --datasets DATASETS    Comma-separated list of datasets (default: all)
  -r, --root-dir DIR         Root directory for data (default: current directory)
  -s, --supervisions         Extract supervisions to JSON files
  -v, --verbose              Enable verbose logging
  -h, --help                 Show this help message

Available single-mic datasets: ${AVAILABLE_DATASETS[*]}

Examples:
  $SCRIPT_NAME -d librimix -s -r /path/to/data
  $SCRIPT_NAME --datasets all --supervisions --root-dir /data
EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--datasets)
                [[ -n "${2:-}" ]] || { log_error "Dataset list cannot be empty"; exit 1; }
                DATASETS="$2"
                shift 2
                ;;
            -r|--root-dir)
                [[ -n "${2:-}" ]] || { log_error "Root directory cannot be empty"; exit 1; }
                ROOT_DIR="$2"
                shift 2
                ;;
            -s|--supervisions)
                EXTRACT_SUPERVISIONS=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Logging functions
log_info() {
    echo "[INFO] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

log_debug() {
    [[ "$VERBOSE" == true ]] && echo "[DEBUG] $*" >&2
}

# Setup directory structure
setup_directories() {
    local data_dir="$ROOT_DIR/data"
    local manifests_dir="$ROOT_DIR/manifests"

    log_info "Setting up directory structure..."
    mkdir -p "$data_dir" "$manifests_dir" "$data_dir/tmp" "$DATASET_SCRIPTS_DIR"

    # Export for use in other functions
    export DATA_DIR="$data_dir"
    export MANIFESTS_DIR="$manifests_dir"
    export DATA_SCRIPTS_PATH="$SCRIPT_DIR/src"
}

# Check if dataset is available
is_dataset_available() {
    local dataset="$1"
    local available_dataset

    for available_dataset in "${AVAILABLE_DATASETS[@]}"; do
        [[ "$available_dataset" == "$dataset" ]] && return 0
    done
    return 1
}

# Check if dataset dependency is satisfied
check_dependency() {
    local dataset="$1"
    local dependency="$(get_dataset_dependency "$dataset")"

    if [[ -n "$dependency" ]]; then
        local dep_manifest="$MANIFESTS_DIR/${dependency}/${dependency}_cutset_train-clean-100.jsonl.gz"
        if [[ ! -f "$dep_manifest" ]]; then
            log_error "Dependency '$dependency' not found for dataset '$dataset'"
            log_error "Please prepare '$dependency' first"
            return 1
        fi
        log_info "Dependency '$dependency' satisfied for dataset '$dataset'"
    fi
    return 0
}

# Prepare a single dataset
prepare_dataset() {
    local dataset="$1"

    # Check dependencies first
    check_dependency "$dataset" || return 1

    # Handle AMI datasets specially
    if [[ "$dataset" =~ ^ami- ]]; then
        local mic_type="${dataset#ami-}"  # Extract mic type from dataset name
        log_info "Preparing AMI dataset: $dataset (mic: $mic_type)"
        log_debug "Running AMI script with mic type: $mic_type"

        if bash "$DATASET_SCRIPTS_DIR/prepare_ami.sh" "$DATA_DIR" "$MANIFESTS_DIR" "$DATA_SCRIPTS_PATH" "$mic_type"; then
            log_info "Completed dataset: $dataset"
        else
            log_error "Failed to prepare dataset: $dataset"
            return 1
        fi
    elif [[ "$dataset" == "ali_meeting-sdm" ]]; then
        log_info "Preparing NotSoFar1 SDM dataset"
        log_debug "Running NotSoFar script with mic type: sdm"

        if bash "$DATASET_SCRIPTS_DIR/prepare_ali_meeting.sh" "$DATA_DIR" "$MANIFESTS_DIR" "$DATA_SCRIPTS_PATH" "sdm"; then
            log_info "Completed dataset: $dataset"
        else
            log_error "Failed to prepare dataset: $dataset"
            return 1
        fi
    # Handle NotSoFar1 SDM dataset specially
    elif [[ "$dataset" == "notsofar1-sdm" ]]; then
        log_info "Preparing NotSoFar1 SDM dataset"
        log_debug "Running NotSoFar script with mic type: sdm"

        if bash "$DATASET_SCRIPTS_DIR/prepare_notsofar.sh" "$DATA_DIR" "$MANIFESTS_DIR" "$DATA_SCRIPTS_PATH" "sdm"; then
            log_info "Completed dataset: $dataset"
        else
            log_error "Failed to prepare dataset: $dataset"
            return 1
        fi
    else
        # Standard dataset preparation - convert hyphens to underscores for script names
        local script_name="prepare_${dataset//-/_}.sh"
        local script_path="$DATASET_SCRIPTS_DIR/$script_name"

        if [[ ! -f "$script_path" ]]; then
            log_error "Dataset script not found: $script_path"
            return 1
        fi

        log_info "Preparing dataset: $dataset"
        log_debug "Running script: $script_path"

        if bash "$script_path" "$DATA_DIR" "$MANIFESTS_DIR" "$DATA_SCRIPTS_PATH"; then
            log_info "Completed dataset: $dataset"
        else
            log_error "Failed to prepare dataset: $dataset"
            return 1
        fi
    fi
}

# Validate dataset list
validate_datasets() {
    local -a datasets_to_validate
    local dataset

    if [[ "$DATASETS" == "all" ]]; then
        return 0
    fi

    IFS=',' read -ra datasets_to_validate <<< "$DATASETS"
    for dataset in "${datasets_to_validate[@]}"; do
        dataset="$(echo "$dataset" | xargs)"  # trim whitespace
        if ! is_dataset_available "$dataset"; then
            log_error "Unknown dataset: '$dataset'"
            log_error "Available datasets: ${AVAILABLE_DATASETS[*]}"
            return 1
        fi
    done
}

# Display configuration
show_configuration() {
    log_info "Single-mic Dataset Preparation Configuration"
    log_info "Datasets: $DATASETS"
    log_info "Extract supervisions: $EXTRACT_SUPERVISIONS"
    log_info "Data directory: $DATA_DIR"
    log_info "Manifests directory: $MANIFESTS_DIR"
    log_info "Verbose mode: $VERBOSE"
}

# Extract supervisions for single-mic datasets
extract_supervisions() {
    log_info "Extracting single-mic supervisions to JSON files"

    # Single-channel supervision mappings
    declare -A sc_files=(
        ["ami-sdm_supervisions_test.jsonl.gz"]="ami-sdm.jsonl.gz"
        ["ami-ihm-mix_supervisions_test.jsonl.gz"]="ami-ihm-mix.jsonl.gz"
        ["notsofar1-sdm_supervisions_eval_sc.jsonl.gz"]="notsofar1-small-sdm.jsonl.gz"
        ["libri2mix_mix_clean_sc_test_supervisions.jsonl.gz"]="libri2mix_clean.jsonl.gz"
        ["libri2mix_mix_both_sc_test_supervisions.jsonl.gz"]="libri2mix_both.jsonl.gz"
        ["libri3mix_mix_clean_sc_test_supervisions.jsonl.gz"]="libri3mix_clean.jsonl.gz"
        ["librispeechmix_test-clean-1mix_supervisions.jsonl.gz"]="librispeechmix_test-clean-1mix.jsonl.gz"
        ["librispeechmix_test-clean-2mix_supervisions.jsonl.gz"]="librispeechmix_test-clean-2mix.jsonl.gz"
        ["librispeechmix_test-clean-3mix_supervisions.jsonl.gz"]="librispeechmix_test-clean-3mix.jsonl.gz"
    )

    # Process single-channel supervisions
    mkdir -p "$DATA_DIR/manifests_sups_test_sc" "$DATA_DIR/refs_test_sc"

    local source_file dest_file source_path dest_path
    for source_file in "${!sc_files[@]}"; do
        dest_file="${sc_files[$source_file]}"
        source_path="$MANIFESTS_DIR/$source_file"
        dest_path="$DATA_DIR/manifests_sups_test_sc/$dest_file"

        if [[ -f "$source_path" ]]; then
            log_debug "Copying: $source_path -> $dest_path"
            cp "$source_path" "$dest_path"
        fi
    done

    # Convert to JSON
    for input_file in "$DATA_DIR/manifests_sups_test_sc"/*.jsonl.gz; do
        if [[ -f "$input_file" ]]; then
            filename="$(basename "$input_file")"
            output_filename="${filename/.jsonl.gz/.json}"
            output_file="$DATA_DIR/refs_test_sc/$output_filename"

            log_debug "Converting: $input_file -> $output_file"
            python "$DATA_SCRIPTS_PATH/supervision_to_hyp_json.py" \
                --input "$input_file" --output "$output_file"
        fi
    done

    log_info "Single-mic supervision extraction completed"
}

# Prepare all requested datasets with dependency resolution
prepare_datasets() {
    local -a datasets_to_prepare
    local dataset

    if [[ "$DATASETS" == "all" ]]; then
        datasets_to_prepare=("${AVAILABLE_DATASETS[@]}")
    else
        IFS=',' read -ra datasets_to_prepare <<< "$DATASETS"
    fi

    # Sort datasets to handle dependencies
    local -a sorted_datasets=()
    for dataset in "${datasets_to_prepare[@]}"; do
        dataset="$(echo "$dataset" | xargs)"

        # Add dependency first if needed
        local dep="$(get_dataset_dependency "$dataset")"
        if [[ -n "$dep" ]]; then
            if [[ ! " ${sorted_datasets[*]-} " =~ " ${dep} " ]]; then
                sorted_datasets+=("$dep")
            fi
        fi

        # Add the dataset itself
        if [[ ! " ${sorted_datasets[*]-} " =~ " ${dataset} " ]]; then
            sorted_datasets+=("$dataset")
        fi
    done

    for dataset in "${sorted_datasets[@]}"; do
        prepare_dataset "$dataset" || {
            log_error "Dataset preparation failed: $dataset"
            return 1
        }
    done
}

# Main execution function
main() {
    parse_arguments "$@"

    # Validate inputs
    validate_datasets || exit 1

    # Setup environment
    setup_directories
    show_configuration

    # Execute main tasks
    prepare_datasets || {
        log_error "Dataset preparation failed"
        exit 1
    }

    if [[ "$EXTRACT_SUPERVISIONS" == true ]]; then
        extract_supervisions || {
            log_error "Supervision extraction failed"
            exit 1
        }
    fi

    log_info "All single-mic dataset preparation completed successfully"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
