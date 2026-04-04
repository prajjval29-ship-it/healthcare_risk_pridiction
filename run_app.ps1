# Starts the API and Streamlit in two new PowerShell windows.
# Run from this folder:  .\run_app.ps1
$ErrorActionPreference = "Stop"
$appDir = $PSScriptRoot
$rootDir = Split-Path $appDir -Parent
$py = Join-Path $rootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$apiCmd = "Set-Location '$appDir'; & '$py' -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8010"
$uiCmd = "Set-Location '$appDir'; `$env:HEALTHCARE_API_URL='http://127.0.0.1:8010'; & '$py' -m streamlit run streamlit_app.py"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCmd
Start-Sleep -Seconds 2
Start-Process powershell -ArgumentList "-NoExit", "-Command", $uiCmd
Write-Host "Opened two windows: API http://127.0.0.1:8010/docs  |  Streamlit http://localhost:8501"
