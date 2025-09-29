#!/bin/bash

# Multi-microphone dataset preparation script
# Usage: ./prepare_multi_mic.sh [OPTIONS]

set -euo pipefail

# Constants
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
readonly DATASET_SCRIPTS_DIR="$SCRIPT_DIR/dataset_scripts"

# Available multi-mic datasets
readonly AVAILABLE_DATASETS=(
    "ami-mdm" "notsofar1-mdm"
)

# Default configuration
DATASETS="all"
EXTRACT_SUPERVISIONS=false
ROOT_DIR="$(pwd)"
VERBOSE=false

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

Available multi-mic datasets: ${AVAILABLE_DATASETS[*]}

Examples:
  $SCRIPT_NAME -d ami-mdm,chime6 -s -r /path/to/data
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

# Check if dataset is available
is_dataset_available() {
    local dataset="$1"
    local available_dataset

    for available_dataset in "${AVAILABLE_DATASETS[@]}"; do
        [[ "$available_dataset" == "$dataset" ]] && return 0
    done
    return 1
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

# Display configuration
show_configuration() {
    log_info "Multi-mic Dataset Preparation Configuration"
    log_info "Datasets: $DATASETS"
    log_info "Extract supervisions: $EXTRACT_SUPERVISIONS"
    log_info "Data directory: $DATA_DIR"
    log_info "Manifests directory: $MANIFESTS_DIR"
    log_info "Verbose mode: $VERBOSE"
}

# Prepare a single dataset
prepare_dataset() {
    local dataset="$1"

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
    # Handle NotSoFar1 MDM dataset specially
    elif [[ "$dataset" == "notsofar1-mdm" ]]; then
        log_info "Preparing NotSoFar1 MDM dataset"
        log_debug "Running NotSoFar script with mic type: mdm"

        if bash "$DATASET_SCRIPTS_DIR/prepare_notsofar.sh" "$DATA_DIR" "$MANIFESTS_DIR" "$DATA_SCRIPTS_PATH" "mdm"; then
            log_info "Completed dataset: $dataset"
        else
            log_error "Failed to prepare dataset: $dataset"
            return 1
        fi
    else
        # Standard dataset preparation
        local script_path="$DATASET_SCRIPTS_DIR/prepare_${dataset//-/_}.sh"

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

# Prepare all requested datasets
prepare_datasets() {
    local -a datasets_to_prepare
    local dataset

    if [[ "$DATASETS" == "all" ]]; then
        datasets_to_prepare=("${AVAILABLE_DATASETS[@]}")
    else
        IFS=',' read -ra datasets_to_prepare <<< "$DATASETS"
    fi

    for dataset in "${datasets_to_prepare[@]}"; do
        dataset="$(echo "$dataset" | xargs)"  # trim whitespace
        prepare_dataset "$dataset" || {
            log_error "Dataset preparation failed: $dataset"
            return 1
        }
    done
}

# Extract supervisions for multi-mic datasets
extract_supervisions() {
    log_info "Extracting multi-mic supervisions to JSON files"

    # Multi-channel supervision mappings
    declare -A mc_files=(
        ["ami-mdm_supervisions_test.jsonl.gz"]="ami-mdm.jsonl.gz"
        ["notsofar1-mdm_supervisions_eval.jsonl.gz"]="notsofar1-small-mdm.jsonl.gz"
    )

    # Process multi-channel supervisions
    mkdir -p "$DATA_DIR/manifests_sups_test_mc" "$DATA_DIR/refs_test_mc"

    local source_file dest_file source_path dest_path
    for source_file in "${!mc_files[@]}"; do
        dest_file="${mc_files[$source_file]}"
        source_path="$MANIFESTS_DIR/$source_file"
        dest_path="$DATA_DIR/manifests_sups_test_mc/$dest_file"

        if [[ -f "$source_path" ]]; then
            log_debug "Copying: $source_path -> $dest_path"
            cp "$source_path" "$dest_path"
        fi
    done

    # Convert to JSON
    for input_file in "$DATA_DIR/manifests_sups_test_mc"/*.jsonl.gz; do
        if [[ -f "$input_file" ]]; then
            filename="$(basename "$input_file")"
            output_filename="${filename/.jsonl.gz/.json}"
            output_file="$DATA_DIR/refs_test_mc/$output_filename"

            log_debug "Converting: $input_file -> $output_file"
            python3 "$DATA_SCRIPTS_PATH/supervision_to_hyp_json.py" \
                --input "$input_file" --output "$output_file"
        fi
    done

    log_info "Multi-mic supervision extraction completed"
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

    log_info "All multi-mic dataset preparation completed successfully"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
