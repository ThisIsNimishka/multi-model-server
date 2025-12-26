# ============================================
# Flash Attention 2 - Check & Install Script
# ============================================
# Run this in your vllm_env virtual environment
# Usage: .\setup_flash_attention.ps1
# ============================================

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Flash Attention 2 - Setup Script          " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ============================================
# Step 1: Check Python Version
# ============================================
Write-Host "[STEP 1] Checking Python Version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  $pythonVersion" -ForegroundColor White

$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    
    if ($major -eq 3 -and $minor -ge 10 -and $minor -le 11) {
        Write-Host "  [OK] Python version compatible (3.10-3.11 required)" -ForegroundColor Green
    } else {
        Write-Host "  [WARNING] Python $major.$minor detected. Flash Attention works best with 3.10 or 3.11" -ForegroundColor Yellow
    }
}
Write-Host ""

# ============================================
# Step 2: Check CUDA Version
# ============================================
Write-Host "[STEP 2] Checking CUDA Version..." -ForegroundColor Yellow

$cudaCheck = python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')" 2>&1
Write-Host "  $cudaCheck" -ForegroundColor White

$nvccVersion = nvcc --version 2>&1 | Select-String "release" | ForEach-Object { $_.Line }
if ($nvccVersion) {
    Write-Host "  NVCC: $nvccVersion" -ForegroundColor White
}
Write-Host ""

# ============================================
# Step 3: Check GPU Compatibility
# ============================================
Write-Host "[STEP 3] Checking GPU Compatibility..." -ForegroundColor Yellow

$gpuCheck = python -c @"
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    arch = f'{props.major}.{props.minor}'
    compatible = props.major >= 8  # Ampere (SM 8.0) or newer
    status = 'Compatible' if compatible else 'Not Compatible'
    print(f'  GPU {i}: {props.name} (SM {arch}) - {status}')
"@ 2>&1

Write-Host $gpuCheck -ForegroundColor White
Write-Host "  [INFO] Flash Attention 2 requires Ampere (RTX 30xx) or newer GPUs" -ForegroundColor Gray
Write-Host ""

# ============================================
# Step 4: Check if Flash Attention is Installed
# ============================================
Write-Host "[STEP 4] Checking Flash Attention Installation..." -ForegroundColor Yellow

$flashCheck = python -c @"
try:
    import flash_attn
    print(f'  [INSTALLED] flash-attn version: {flash_attn.__version__}')
except ImportError:
    print('  [NOT INSTALLED] flash-attn is not installed')
"@ 2>&1

Write-Host $flashCheck -ForegroundColor White
Write-Host ""

# ============================================
# Step 5: Check Transformers Flash Attention Support
# ============================================
Write-Host "[STEP 5] Checking Transformers Flash Attention Support..." -ForegroundColor Yellow

$transformersCheck = python -c @"
import transformers
print(f'  Transformers version: {transformers.__version__}')

# Check if version supports flash attention
version_parts = transformers.__version__.split('.')
major = int(version_parts[0])
minor = int(version_parts[1]) if len(version_parts) > 1 else 0

if major >= 4 and minor >= 34:
    print('  [OK] Transformers version supports Flash Attention 2')
else:
    print('  [WARNING] Update transformers: pip install transformers>=4.34.0')
"@ 2>&1

Write-Host $transformersCheck -ForegroundColor White
Write-Host ""

# ============================================
# Step 6: Installation Options
# ============================================
Write-Host "[STEP 6] Installation Options" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Option A: Install from pip (may need compilation)" -ForegroundColor Cyan
Write-Host "  pip install flash-attn --no-build-isolation" -ForegroundColor White
Write-Host ""
Write-Host "  Option B: Install pre-built wheel (faster)" -ForegroundColor Cyan
Write-Host "  pip install flash-attn --no-build-isolation" -ForegroundColor White
Write-Host ""
Write-Host "  Option C: Use SDPA (built into PyTorch 2.0+, no install needed)" -ForegroundColor Cyan
Write-Host "  This is enabled by default in newer transformers" -ForegroundColor Gray
Write-Host ""

# ============================================
# Step 7: Ask to Install
# ============================================
Write-Host "============================================" -ForegroundColor DarkGray
$install = Read-Host "Do you want to try installing flash-attn now? (y/n)"

if ($install -eq 'y' -or $install -eq 'Y') {
    Write-Host ""
    Write-Host "Installing flash-attn..." -ForegroundColor Yellow
    Write-Host "This may take several minutes..." -ForegroundColor Gray
    Write-Host ""
    
    # Try pip install
    pip install flash-attn --no-build-isolation
    
    Write-Host ""
    Write-Host "Verifying installation..." -ForegroundColor Yellow
    
    python -c @"
try:
    import flash_attn
    print(f'SUCCESS! flash-attn {flash_attn.__version__} installed')
except ImportError as e:
    print(f'Installation may have failed: {e}')
    print('Try using SDPA instead (built into PyTorch)')
"@
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!                           " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
