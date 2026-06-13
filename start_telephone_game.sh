#!/bin/bash
# start_telephone_game.sh
# Start all backend processes for the telephone game.
# Run this first, then run: uv run pbm_stream_to_ffmpeg.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Starting Telephone Game Backend ==="

# 1. Start NATS server
echo "[1/4] Starting NATS server..."
./nats-server -c server.config &
NATS_PID=$!
sleep 1

# 2. Start ws_bridge (subject "two" — the classifier output)
echo "[2/4] Starting ws_bridge on subject 'two'..."
cd publisher
uv run ws_bridge.py two &
BRIDGE_PID=$!
cd "$SCRIPT_DIR"
sleep 1

# 3. Start publisher (generates digits → NATS "one")
echo "[3/4] Starting publisher..."
cd publisher
uv run publish.py &
PUBLISHER_PID=$!
cd "$SCRIPT_DIR"
sleep 1

# 4. Start classifier (classifies "one" → generates → NATS "two")
echo "[4/4] Starting classifier..."
uv run classifier_cvae.py &
CLASSIFIER_PID=$!

echo ""
echo "=== All processes started ==="
echo "  NATS server:     PID $NATS_PID"
echo "  ws_bridge:       PID $BRIDGE_PID"
echo "  publisher:       PID $PUBLISHER_PID"
echo "  classifier:      PID $CLASSIFIER_PID"
echo ""
echo "Now run: uv run pbm_stream_to_ffmpeg.py"
echo ""
echo "Press Ctrl+C to stop all processes."

# Wait for Ctrl+C, then clean up all processes
trap "echo 'Stopping...'; kill $CLASSIFIER_PID $PUBLISHER_PID $BRIDGE_PID $NATS_PID 2>/dev/null; wait; exit" INT TERM

wait
