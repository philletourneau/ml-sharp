$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[build] $msg" }

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path

$repoRoot = (Get-Location).Path
$venv = Join-Path $repoRoot ".venv-build-photo-to-viewer-gui"
$assetsDir = Join-Path $repoRoot "packaging\\windows\\assets"
$nodeDir = Join-Path $assetsDir "node"
$vendorDir = Join-Path $assetsDir "splat-transform"

$splatVersion = $env:SPLAT_TRANSFORM_VERSION
if (-not $splatVersion) { $splatVersion = "0.16.0" }

New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null

if (-not (Test-Path (Join-Path $nodeDir "node.exe"))) {
    Write-Info "Downloading Node.js (latest LTS) for bundling"
    $index = Invoke-RestMethod -Uri "https://nodejs.org/dist/index.json"
    $lts = $index | Where-Object { $_.lts -ne $false } | Select-Object -First 1
    if (-not $lts) { throw "Could not determine latest Node LTS version." }
    $nodeVersion = $lts.version  # e.g. v20.18.1
    $zipName = "node-$nodeVersion-win-x64.zip"
    $zipUrl = "https://nodejs.org/dist/$nodeVersion/$zipName"
    $zipPath = Join-Path $assetsDir $zipName

    Write-Info "Downloading $zipUrl"
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

    Write-Info "Extracting Node to assets/"
    Expand-Archive -Path $zipPath -DestinationPath $assetsDir -Force

    $extracted = Join-Path $assetsDir ("node-$nodeVersion-win-x64")
    if (-not (Test-Path $extracted)) { throw "Expected extracted folder not found: $extracted" }
    if (Test-Path $nodeDir) { Remove-Item -Recurse -Force $nodeDir }
    Move-Item -Path $extracted -Destination $nodeDir
}

$npm = Join-Path $nodeDir "npm.cmd"
if (-not (Test-Path $npm)) { throw "npm.cmd not found in bundled Node at $npm" }

# Ensure postinstall scripts can find `node` on PATH.
$env:PATH = "$nodeDir;$env:PATH"

Write-Info "Installing @playcanvas/splat-transform@$splatVersion into assets/"
New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
& $npm install --prefix $vendorDir --no-audit --no-fund ("@playcanvas/splat-transform@$splatVersion")
if ($LASTEXITCODE -ne 0) { throw "npm install failed." }

if (-not (Test-Path $venv)) {
    Write-Info "Creating build venv at $venv"
    py -3.12 -m venv $venv
}
else {
    Write-Info "Using existing build venv at $venv"
}

$py = Join-Path $venv "Scripts\\python.exe"
& $py -m pip install --upgrade pip setuptools wheel

if (-not (Test-Path (Join-Path $venv "Lib\\site-packages\\torch"))) {
    Write-Info "Installing CUDA-enabled PyTorch into build venv"
    & $py -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
}
else {
    Write-Info "CUDA-enabled PyTorch already present in build venv"
}

Write-Info "Installing project + build tools"
& $py -m pip install -e .
& $py -m pip install pyinstaller ninja

if (-not (Test-Path (Join-Path $assetsDir "gsplat_cuda.pyd"))) {
    Write-Info "Patching gsplat Windows compile flags in build venv (workaround)"
    & $py -c @"
import gsplat
from pathlib import Path
import re

p = Path(gsplat.__file__).parent / "cuda" / "_backend.py"
txt = p.read_text(encoding="utf-8")
changed = False
# Remove GCC-only flag that breaks MSVC `cl.exe`.
txt2 = re.sub(r',\s*"-Wno-attributes"', "", txt)
if txt2 != txt:
    txt = txt2
    changed = True
if changed:
    p.write_text(txt, encoding="utf-8")
    print("patched", p)
else:
    print("no patch needed", p)
"@

    Write-Info "Locating Visual Studio Build Tools"
    $vswhere = "${env:ProgramFiles(x86)}\\Microsoft Visual Studio\\Installer\\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found; install Visual Studio Build Tools 2022." }
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) { throw "Visual Studio with VC tools not found." }
    $vcvars = Join-Path $vsPath "VC\\Auxiliary\\Build\\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }

    Write-Info "Building gsplat CUDA extension (one-time, for bundling)"
    cmd.exe /c "`"$vcvars`" && `"$py`" -c `"import gsplat.cuda._backend`""
    if ($LASTEXITCODE -ne 0) { throw "Failed to build/import gsplat CUDA backend." }

    Write-Info "Copying built gsplat extension into assets/"
    & $py -c @"
from pathlib import Path
from torch.utils.cpp_extension import _get_build_directory
build_dir = Path(_get_build_directory('gsplat_cuda', verbose=False))
src = build_dir / 'gsplat_cuda.pyd'
dst = Path(r'$assetsDir') / 'gsplat_cuda.pyd'
print('src', src)
print('dst', dst)
dst.write_bytes(src.read_bytes())
"@
    if ($LASTEXITCODE -ne 0) { throw "Failed to copy gsplat_cuda.pyd into assets." }
    if (-not (Test-Path (Join-Path $assetsDir "gsplat_cuda.pyd"))) { throw "assets\\gsplat_cuda.pyd not found after copy." }
}
else {
    Write-Info "Using existing assets\\gsplat_cuda.pyd"
}

Write-Info "Running PyInstaller"
& $py -m PyInstaller --clean --noconfirm packaging\\windows\\sharp-photo-to-viewer-gui.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

Write-Info "Done. Output: dist\\sharp-photo-to-viewer-gui\\sharp-photo-to-viewer-gui.exe"
