# Model Decay Radar - Windows PowerShell Startup Script

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "       MODEL DECAY RADAR v2.0             " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

# 1. Clean up old processes on port 8000 and 3000
Write-Host "[*] Checking and freeing ports 8000 and 3000..." -ForegroundColor Yellow
$oldProcesses = Get-NetTCPConnection -LocalPort 8000, 3000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($pidToKill in $oldProcesses) {
    Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
}

# Ensure logs directory exists
if (-not (Test-Path "$ROOT/logs")) {
    New-Item -ItemType Directory -Path "$ROOT/logs" | Out-Null
}

# 2. Check dataset
if (-not (Test-Path "$ROOT/data/synthetic_swat.csv")) {
    Write-Host "[0/2] Generating synthetic SWaT dataset..." -ForegroundColor Yellow
    $env:PYTHONPATH = "$ROOT/src;$ROOT/pipeline;$ROOT"
    python "$ROOT/src/generate_synthetic_swat.py"
    Write-Host "      Dataset ready." -ForegroundColor Green
}

# 3. Start FastAPI Server
Write-Host "[1/2] Starting FastAPI Backend on http://localhost:8000 ..." -ForegroundColor Cyan
$apiProcess = Start-Process -FilePath "python" `
    -ArgumentList "-m uvicorn api.server:app --host 127.0.0.1 --port 8000" `
    -WorkingDirectory $ROOT `
    -Environment @{ PYTHONPATH = "$ROOT/src;$ROOT/pipeline;$ROOT" } `
    -PassThru -NoNewWindow `
    -RedirectStandardOutput "$ROOT/logs/api.log" `
    -RedirectStandardError "$ROOT/logs/api_err.log"

Write-Host "      FastAPI Process ID: $($apiProcess.Id)" -ForegroundColor Green

# 4. Start Next.js Frontend
Write-Host "[2/2] Starting Next.js Frontend on http://localhost:3000 ..." -ForegroundColor Cyan
$frontProcess = Start-Process -FilePath "npm.cmd" `
    -ArgumentList "run dev" `
    -WorkingDirectory "$ROOT/frontend" `
    -PassThru -NoNewWindow `
    -RedirectStandardOutput "$ROOT/logs/frontend.log" `
    -RedirectStandardError "$ROOT/logs/frontend_err.log"

Write-Host "      Next.js Process ID: $($frontProcess.Id)" -ForegroundColor Green

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host " System is initializing in the background! " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard URL : http://localhost:3000" -ForegroundColor White
Write-Host "  API Docs URL  : http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "  NOTE: Initializing the VAE and RNN takes ~30-60 seconds on boot." -ForegroundColor Yellow
Write-Host "  Open http://localhost:3000 in your browser." -ForegroundColor Yellow
Write-Host ""
