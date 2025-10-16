#!/bin/bash
# Helper script to open multiple terminal windows to monitor debates
# Usage: ./monitor_debates.sh debate_1_*.log debate_2_*.log

if [ $# -eq 0 ]; then
    echo "Usage: $0 <log_file1> <log_file2> [log_file3...]"
    echo ""
    echo "Example:"
    echo "  $0 test_debate_results/debate_1_*.log test_debate_results/debate_2_*.log"
    exit 1
fi

# macOS - open new Terminal windows
if [[ "$OSTYPE" == "darwin"* ]]; then
    for log_file in "$@"; do
        # Expand glob pattern
        for file in $log_file; do
            if [ -f "$file" ]; then
                osascript -e "tell application \"Terminal\" to do script \"tail -f $PWD/$file\""
            fi
        done
    done
# Linux with gnome-terminal
elif command -v gnome-terminal &> /dev/null; then
    for log_file in "$@"; do
        for file in $log_file; do
            if [ -f "$file" ]; then
                gnome-terminal -- bash -c "tail -f $PWD/$file; exec bash"
            fi
        done
    done
# Linux with xterm
elif command -v xterm &> /dev/null; then
    for log_file in "$@"; do
        for file in $log_file; do
            if [ -f "$file" ]; then
                xterm -e "tail -f $PWD/$file" &
            fi
        done
    done
else
    echo "Could not detect terminal application."
    echo "Please manually run: tail -f <log_file>"
fi

