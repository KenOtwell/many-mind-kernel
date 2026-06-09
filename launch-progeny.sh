#!/usr/bin/env bash
# Launch the MMK Progeny stack: llama-server (Mistral) + Progeny FastAPI service
#
# Usage:  ./launch-progeny.sh [--qdrant-host IP]
#
# Defaults:
#   Qdrant:  192.168.0.13 (StealthVI)
#   LLM:     Mistral Nemo 12B Q8 on llama-server :8080
#   Progeny: FastAPI on :8001

set -euo pipefail

QDRANT_HOST="${1:-192.168.0.13}"
LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
MODEL="$HOME/models/gguf/mistral-nemo-12b-instruct-q8.gguf"
VENV="$HOME/Neo/.venv/bin/python"
PROJECT="$HOME/Neo"
LOG_DIR="$HOME/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"
LLAMA_LOG="$LOG_DIR/llama-server.log"
PROGENY_LOG="$LOG_DIR/progeny.log"

echo "=== MMK Progeny Stack ==="
echo "  Qdrant:  $QDRANT_HOST:6333"
echo "  LLM:     Mistral Nemo 12B → :8080"
echo "  Progeny: FastAPI → :8001"
echo ""

# --- Kill any existing instances ---
pkill -f "llama-server.*8080" 2>/dev/null || true
pkill -f "uvicorn progeny.api.server" 2>/dev/null || true
sleep 1

# --- Launch llama-server ---
echo "[1/2] Starting llama-server... (log: $LLAMA_LOG)"
"$LLAMA_SERVER" \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 99 \
    -c 8192 \
    --no-mmap \
    >> "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
echo "  llama-server PID: $LLAMA_PID"

# Wait for llama-server to be ready
echo "  Waiting for LLM to load..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
        echo "  LLM ready."
        break
    fi
    sleep 2
done

# --- Launch Progeny ---
echo "[2/2] Starting Progeny... (log: $PROGENY_LOG)"
cd "$PROJECT"
QDRANT_HOST="$QDRANT_HOST" \
LLM_PROFILE=mistral-nemo \
LLM_HOST=127.0.0.1 \
LLM_PORT=8080 \
"$VENV" -m uvicorn progeny.api.server:app \
    --host 0.0.0.0 \
    --port 8001 \
    >> "$PROGENY_LOG" 2>&1 &
PROGENY_PID=$!
echo "  Progeny PID: $PROGENY_PID"

sleep 3

# --- Health check ---
echo ""
echo "=== Health Check ==="
curl -s http://127.0.0.1:8001/health 2>/dev/null || echo "  Progeny not responding yet (may still be loading)"
echo ""
echo ""
echo "=== Stack Running ==="
echo "  llama-server: PID $LLAMA_PID (port 8080)"
echo "  Progeny:      PID $PROGENY_PID (port 8001)"
echo ""
echo "To stop: kill $LLAMA_PID $PROGENY_PID"
echo "Or:      pkill -f llama-server; pkill -f 'uvicorn progeny'"
echo ""
echo "Logs:"
echo "  tail -f $PROGENY_LOG"
echo "  tail -f $LLAMA_LOG"

# Keep script alive so both background processes stay running
wait
