# Model Decay Radar - Windows PowerShell Stop Script

Write-Host "[*] Stopping Model Decay Radar background services..." -ForegroundColor Yellow

$ports = @(8000, 3000)
$pids = Get-NetTCPConnection -LocalPort $ports -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique

if ($pids) {
    foreach ($p in $pids) {
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        Write-Host "    Terminated PID: $p" -ForegroundColor Green
    }
    Write-Host "All Model Decay Radar services stopped." -ForegroundColor Green
} else {
    Write-Host "No active services found on ports 8000 or 3000." -ForegroundColor Cyan
}
