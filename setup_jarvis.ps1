# JARVIS setup script (Windows PowerShell)
# Creates the dedicated Jarvis virtual environment (Python 3.13) and installs
# CPU-only PyTorch + the modular requirement sets. Does NOT touch the YAKI venv.
# Usage: powershell -ExecutionPolicy Bypass -File setup_jarvis.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host "== JARVIS setup ==" -ForegroundColor Cyan

# 1. Python interpreter (3.13 needed: torch has no 3.14 wheels)
$py = (py -3.13 -c "import sys; print(sys.executable)") 2>$null
if (-not $py) {
    Write-Host "FAIL: Python 3.13 not found via py launcher. Install Python 3.13 from python.org" -ForegroundColor Red
    exit 1
}
Write-Host "Using Python: $py"

# 2. venv
if (-not (Test-Path "$Root\venv")) {
    py -3.13 -m venv "$Root\venv"
    Write-Host "Created venv\"
} else {
    Write-Host "venv\ already exists (keeping it)"
}

$vpy = "$Root\venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip --quiet

# 3. CPU-only PyTorch first (Windows, no CUDA on this hardware)
& $vpy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
& $vpy -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 4. Modular dependency sets
& $vpy -m pip install -r requirements-core.txt
& $vpy -m pip install -r requirements-voice.txt
& $vpy -m pip install -r requirements-ml.txt
& $vpy -m pip install -r requirements-automation.txt
& $vpy -m pip install -r requirements-dev.txt

# 5. Health check
& $vpy doctor.py

Write-Host ""
Write-Host "Next steps:"
Write-Host "  Models:        .\venv\Scripts\python.exe download_models.py --selected-only"
Write-Host "  Enroll owner:  .\venv\Scripts\python.exe enroll_owner.py"
Write-Host "  Run JARVIS:    .\run_jarvis.bat"
