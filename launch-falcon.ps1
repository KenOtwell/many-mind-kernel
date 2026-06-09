# Launch Falcon — SKSE-facing backend for Many-Mind Kernel
# Run from any PowerShell terminal on the Gaming PC.

$env:PROGENY_HOST = "192.168.0.220"
$env:PROGENY_PORT = "8001"

Write-Host "Falcon starting - Progeny at ws://$($env:PROGENY_HOST):$($env:PROGENY_PORT)/ws"

python -m uvicorn falcon.api.server:app --host 0.0.0.0 --port 8000 --app-dir C:\Users\Ken\Projects\many-mind-kernel `
  2>&1 | Tee-Object -FilePath ".\falcon-$(Get-Date -Format yyyyMMdd-HHmmss).log"
