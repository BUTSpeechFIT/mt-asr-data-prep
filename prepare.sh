#!/bin/bash

# Main dataset preparation script dispatcher
# Usage: ./prepare.sh [OPTIONS]

set -euo pipefail

# Constants
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Available datasets categorized by microphone type
# readonly SINGLE_MIC_DATASETS=("librispeech" "librimix" "librispeechmix" "ami-sdm" "ami-ihm-mix" "notsofar1-sdm")
readonly SINGLE_MIC_DATASETS=("librispeech" "librimix" "librispeechmix" "ami-sdm" "notsofar1-sdm")
# readonly MULTI_MIC_DATASETS=("ami-mdm" "ali_meeting" "aishell4" "chime6" "notsofar1-mdm")
readonly MULTI_MIC_DATASETS=("ami-mdm" "notsofar1-mdm")

# Default configuration
DATASETS="all"
EXTRACT_SUPERVISIONS=false
ROOT_DIR="$(pwd)"
VERBOSE=false
SINGLE_MIC_ONLY=false
MULTI_MIC_ONLY=false

# Logging functions
log_info() {
    echo "[INFO] $*" >&2
}

log_error() {
    echo "[ERROR] $*" >&2
}

# Usage function
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Options:
  -d, --datasets DATASETS    Comma-separated list of datasets (default: all)
  -r, --root-dir DIR         Root directory for data (default: current directory)
  -s, --supervisions         Extract supervisions to JSON files
  --single-mic-only          Process only single-microphone datasets
  --multi-mic-only           Process only multi-microphone datasets
  -v, --verbose              Enable verbose logging
  -h, --help                 Show this help message

Available single-mic datasets: ${SINGLE_MIC_DATASETS[*]}
Available multi-mic datasets: ${MULTI_MIC_DATASETS[*]}

Examples:
  $SCRIPT_NAME -d ami-sdm,ami-ihm-mix -s -r /path/to/data
  $SCRIPT_NAME --datasets all --supervisions --root-dir /data
  $SCRIPT_NAME --single-mic-only --supervisions
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
            --single-mic-only)
                SINGLE_MIC_ONLY=true
                shift
                ;;
            --multi-mic-only)
                MULTI_MIC_ONLY=true
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

# Determine datasets to process for each microphone type
determine_datasets() {
    local -a single_mic_datasets=()
    local -a multi_mic_datasets=()
    local -a requested_datasets=()
    local dataset

    if [[ "$DATASETS" == "all" ]]; then
        if [[ "$MULTI_MIC_ONLY" != true ]]; then
            single_mic_datasets=("${SINGLE_MIC_DATASETS[@]}")
        fi
        if [[ "$SINGLE_MIC_ONLY" != true ]]; then
            multi_mic_datasets=("${MULTI_MIC_DATASETS[@]}")
        fi
    else
        IFS=',' read -ra requested_datasets <<< "$DATASETS"
        for dataset in "${requested_datasets[@]}"; do
            dataset="$(echo "$dataset" | xargs)"  # trim whitespace

            # Check if it's a single-mic dataset
            if [[ " ${SINGLE_MIC_DATASETS[*]} " =~ " ${dataset} " ]]; then
                if [[ "$MULTI_MIC_ONLY" != true ]]; then
                    single_mic_datasets+=("$dataset")
                fi
            # Check if it's a multi-mic dataset
            elif [[ " ${MULTI_MIC_DATASETS[*]} " =~ " ${dataset} " ]]; then
                if [[ "$SINGLE_MIC_ONLY" != true ]]; then
                    multi_mic_datasets+=("$dataset")
                fi
            else
                log_error "Unknown dataset: '$dataset'"
                log_error "Available datasets: ${SINGLE_MIC_DATASETS[*]} ${MULTI_MIC_DATASETS[*]}"
                exit 1
            fi
        done
    fi

    # Export arrays for use in other functions
    export SINGLE_MIC_TO_PROCESS="${single_mic_datasets[*]-}"
    export MULTI_MIC_TO_PROCESS="${multi_mic_datasets[*]-}"
}

# Build script arguments
build_script_args() {
    local args=()

    args+=("--root-dir" "$ROOT_DIR")

    if [[ "$EXTRACT_SUPERVISIONS" == true ]]; then
        args+=("--supervisions")
    fi

    if [[ "$VERBOSE" == true ]]; then
        args+=("--verbose")
    fi

    echo "${args[@]}"
}

# Main execution function
main() {
    parse_arguments "$@"
    determine_datasets

    local script_args
    script_args=($(build_script_args))

    # Process single-mic datasets if any
    if [[ -n "$SINGLE_MIC_TO_PROCESS" ]]; then
        log_info "Processing single-microphone datasets: $SINGLE_MIC_TO_PROCESS"
        "$SCRIPT_DIR/prepare_single_mic.sh" \
            --datasets "${SINGLE_MIC_TO_PROCESS// /,}" \
            "${script_args[@]}" || {
            log_error "Single-mic dataset preparation failed"
            exit 1
        }
    fi

    # Process multi-mic datasets if any
    if [[ -n "$MULTI_MIC_TO_PROCESS" ]]; then
        log_info "Processing multi-microphone datasets: $MULTI_MIC_TO_PROCESS"
        "$SCRIPT_DIR/prepare_multi_mic.sh" \
            --datasets "${MULTI_MIC_TO_PROCESS// /,}" \
            "${script_args[@]}" || {
            log_error "Multi-mic dataset preparation failed"
            exit 1
        }
    fi

    log_info "All dataset preparation completed successfully"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
