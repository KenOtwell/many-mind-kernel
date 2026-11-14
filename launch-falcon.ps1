# Launch Falcon — SKSE-facing backend for Many-Mind Kernel
# Run from any PowerShell terminal on the Gaming PC.

# Per-machine overrides from a gitignored .env next to this script (see .env.example).
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*#') { return }
        if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2].Trim('"')
        }
    }
}
if (-not $env:PROGENY_HOST) { $env:PROGENY_HOST = "Progeny.kenos.one" }
if (-not $env:PROGENY_PORT) { $env:PROGENY_PORT = "8001" }

Write-Host "Falcon starting - Progeny at ws://$($env:PROGENY_HOST):$($env:PROGENY_PORT)/ws"

python -m uvicorn falcon.api.server:app --host 0.0.0.0 --port 8000 --app-dir C:\Users\Ken\Projects\many-mind-kernel `
  2>&1 | Tee-Object -FilePath ".\falcon-$(Get-Date -Format yyyyMMdd-HHmmss).log"
