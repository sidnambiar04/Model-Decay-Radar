#!/bin/bash
# Model Decay Radar — One-command startup

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Select Python binary that has uvicorn installed
if python -c "import uvicorn" >/dev/null 2>&1; then
  PYTHON=python
elif python3 -c "import uvicorn" >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON=python.exe
else
  PYTHON=python
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║        MODEL DECAY RADAR v2.0            ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Free ports 8000 & 3000 if occupied
echo "[*] Cleaning up existing processes..."
powershell -Command "Get-Process -Name python, node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue" 2>/dev/null || true
pkill -f "uvicorn api.server" 2>/dev/null || true
pkill -f "next dev"           2>/dev/null || true
sleep 1

mkdir -p logs

# Step 1: Generate dataset if missing
if [ ! -f "$ROOT/data/synthetic_swat.csv" ]; then
    echo "[0/2] Generating synthetic SWaT dataset..."
    PYTHONPATH="$ROOT/src:$ROOT/pipeline:$ROOT" "$PYTHON" src/generate_synthetic_swat.py
    echo "      Dataset ready."
    echo ""
fi

# Step 1: Start FastAPI Backend
echo "[1/2] Starting FastAPI server on http://localhost:8000 ..."
PYTHONPATH="$ROOT/src:$ROOT/pipeline:$ROOT" \
  "$PYTHON" -m uvicorn api.server:app \
  --host 127.0.0.1 \
  --port 8000 \
  --log-level info \
  > logs/api.log 2>&1 &
API_PID=$!
echo "      FastAPI PID: $API_PID"
sleep 2

# Step 2: Start Next.js Frontend
echo "[2/2] Starting Next.js frontend dashboard on http://localhost:3000 ..."
(cd "$ROOT/frontend" && npm run dev > "$ROOT/logs/frontend.log" 2>&1) &
FRONT_PID=$!
echo "      Next.js PID: $FRONT_PID"

echo ""
echo "✅ Both services starting up."
echo ""
echo "  📊 Next.js Dashboard : http://localhost:3000"
echo "  ⚡ FastAPI Server    : http://localhost:8000"
echo "  📖 API Documentation : http://localhost:8000/docs"
echo ""
echo "  NOTE: AE + RNN training takes ~1-2 min on startup."
echo "  Open http://localhost:3000 and wait until status shows CONNECTED,"
echo "  then click 'Simulate Drift' in the top bar."
echo ""
