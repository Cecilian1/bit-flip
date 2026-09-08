#!/bin/bash

TARGET="run_causaloneflip.sh main-all"
CHECK_INTERVAL=30
SHUTDOWN_DELAY=120

while pgrep -f "$TARGET" > /dev/null; do
	echo "Still there!"
	sleep "$CHECK_INTERVAL"
done

echo "Calculation finished, shutdown in 120s..."

sleep "$SHUTDOWN_DELAY"

shutdown -h +2
