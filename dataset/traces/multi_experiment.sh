#!/bin/bash

# Usage: ./multi_experiment.sh START STOP SCRIPT_FILENAME TIMEOUT_SECS TARGET_COUNT

START=$1
STOP=$2
SCRIPT_FILENAME=$3
TIMEOUT_SECS=$4
TARGET_COUNT=$5

if [ -z "$START" ] || [ -z "$STOP" ] || [ -z "$SCRIPT_FILENAME" ] || [ -z "$TIMEOUT_SECS" ] || [ -z "$TARGET_COUNT" ]; then
  echo "Usage: $0 START STOP SCRIPT_FILENAME TIMEOUT_SECS TARGET_COUNT"
  exit 1
fi

for (( i=START; i<=STOP; i++ )); do
  echo "Generating data for website $i..."

  # Replace wid="website_i" in the script file
  sed -i "s/wid=\"website_[0-9]\{1,2\}\"/wid=\"website_$i\"/" "$SCRIPT_FILENAME"

  # Run the experiment watchdog
  bash experiment_watchdog.sh "$SCRIPT_FILENAME" "$TIMEOUT_SECS" "$TARGET_COUNT"

  # Change to experiment_data and run compress script
  (
    cd /home/wrana_michael/experiment_data || exit 1
    python3 compress_pcaps.py "$i"

    # Delete contents of output folder (requires sudo)
    sudo rm -rf output/*
    sudo find output -type d ! -path output -exec rm -rf {} +
  )

  echo "Website $i dataset generated."
done