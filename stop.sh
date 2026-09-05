#!/bin/bash
echo "Stopping Model Decay Radar..."
powershell -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_['OwningProcess'] -Force -ErrorAction SilentlyContinue }" 2>/dev/null || true
powershell -Command "Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id \$_['OwningProcess'] -Force -ErrorAction SilentlyContinue }" 2>/dev/null || true
pkill -f "uvicorn api.server" 2>/dev/null || true
pkill -f "next dev"           2>/dev/null || true
echo "  FastAPI & Next.js frontend stopped."
echo "Done."
