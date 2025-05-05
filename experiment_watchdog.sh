#!/bin/bash

# Folder to monitor
WATCH_DIR="/home/michaelw/Documents/experiment_data/output/"
TARGET_COUNT=100
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

    echo "Running new_experiment.py (timeout: 1000s)..."
    
    # Start command in a new process group
    setsid bash -c 'sudo timeout 1000s python3 new_experiment_single.py' &
    child_pid=$!
    wait $child_pid
done