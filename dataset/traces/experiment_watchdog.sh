#!/bin/bash

# Usage: ./experiment_watchdog.sh PYTHON_SCRIPT TIMEOUT_SECS TARGET_COUNT

SCRIPT_NAME=$1
TIMEOUT_SECS=$2
TARGET_COUNT=$3

if [ -z "$SCRIPT_NAME" ] || [ -z "$TIMEOUT_SECS" ] || [ -z "$TARGET_COUNT" ]; then
    echo "Usage: $0 PYTHON_SCRIPT TIMEOUT_SECS TARGET_COUNT"
    exit 1
fi

# Folder to monitor
WATCH_DIR="/home/wrana_michael/experiment_data/output/"
MIN_SIZE=1048576  # 1MB in bytes

# Handle Ctrl+C and forward to subprocess
cleanup() {
    echo "Caught Ctrl+C. Cleaning up..."
    if [[ -n "$child_pid" ]]; then
        echo "Killing child process group $child_pid"
        kill -- -"$child_pid" 2>/dev/null
    fi
    exit 1
}
trap cleanup SIGINT

while true; do
    mapfile -t pcap_files < <(find "$WATCH_DIR" -type f -name "*.pcap" -size +${MIN_SIZE}c)
    file_count="${#pcap_files[@]}"
    
    echo "Found $file_count .pcap files larger than 1MB..."

    if [ "$file_count" -ge "$TARGET_COUNT" ]; then
        echo "Reached target of $TARGET_COUNT files."
        break
    fi

    # Clean /tmp/minindn before each experiment
    if [ -d /tmp/minindn ]; then
        echo "Clearing /tmp/minindn..."
        sudo rm -rf /tmp/minindn/*
        sudo find /tmp/minindn -mindepth 1 -exec rm -rf {} +
    fi

    echo "Running $SCRIPT_NAME (timeout: ${TIMEOUT_SECS}s)..."

    setsid bash -c "sudo timeout ${TIMEOUT_SECS}s python3 $SCRIPT_NAME" &
    child_pid=$!
    wait $child_pid
done