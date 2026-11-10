# Starts one VISTA backend. Refuses to launch if port 8000 is already in use.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listen) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$($listen.OwningProcess)"
    Write-Error "Port 8000 is already in use by PID $($listen.OwningProcess): $($owner.CommandLine)"
}
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 --env-file .env --log-level info --access-log
