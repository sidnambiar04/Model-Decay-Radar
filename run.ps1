# Model Decay Radar — Windows PowerShell One-Command Startup
$ScriptDir = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
Set-Location $ScriptDir

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "        MODEL DECAY RADAR v2.0            " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[*] Stopping any previous server processes on ports 8000 / 3000..." -ForegroundColor Yellow
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

if (-not (Test-Path "data\synthetic_swat.csv")) {
    Write-Host "[0/2] Generating synthetic SWaT dataset..." -ForegroundColor Yellow
    $env:PYTHONPATH = "$ScriptDir\src;$ScriptDir\pipeline;$ScriptDir"
    python src/generate_synthetic_swat.py
    Write-Host "      Dataset ready." -ForegroundColor Green
    Write-Host ""
}

Write-Host "[1/2] Starting FastAPI server on http://localhost:8000 ..." -ForegroundColor Yellow
$env:PYTHONPATH = "$ScriptDir\src;$ScriptDir\pipeline;$ScriptDir"
$apiProcess = Start-Process -FilePath "python" -ArgumentList "-m uvicorn api.server:app --host 127.0.0.1 --port 8000 --log-level info" -PassThru -NoNewWindow
Write-Host "      FastAPI PID: $($apiProcess.Id)" -ForegroundColor Green

Start-Sleep -Seconds 2

Write-Host "[2/2] Starting Next.js frontend dashboard on http://localhost:3000 ..." -ForegroundColor Yellow
$frontendDir = Join-Path $ScriptDir "frontend"
$frontProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c cd /d `"$frontendDir`" && npm run dev" -PassThru -NoNewWindow
Write-Host "      Next.js PID: $($frontProcess.Id)" -ForegroundColor Green

Write-Host ""
Write-Host "[OK] Both services started successfully." -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard URL : http://localhost:3000" -ForegroundColor Cyan
Write-Host "  FastAPI Server: http://localhost:8000" -ForegroundColor Cyan
Write-Host "  API Docs      : http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "  NOTE: AE + RNN training takes ~1 min on startup." -ForegroundColor Yellow
Write-Host "  Open http://localhost:3000 and click 'Simulate Drift' in the top header." -ForegroundColor Yellow
Write-Host ""
