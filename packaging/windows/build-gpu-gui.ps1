$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[build] $msg" }

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path

$repoRoot = (Get-Location).Path
$venv = Join-Path $repoRoot ".venv-build-gui"
$assetsDir = Join-Path $repoRoot "packaging\\windows\\assets"

New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null

Write-Info "Creating build venv at $venv"
py -3.12 -m venv $venv

$py = Join-Path $venv "Scripts\\python.exe"
& $py -m pip install --upgrade pip setuptools wheel

Write-Info "Installing CUDA-enabled PyTorch into build venv"
& $py -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

Write-Info "Installing project + build tools"
& $py -m pip install -e .
& $py -m pip install pyinstaller ninja

Write-Info "Patching gsplat Windows compile flags in build venv (workaround)"
& $py -c @"
import gsplat
from pathlib import Path

p = Path(gsplat.__file__).parent / "cuda" / "_backend.py"
txt = p.read_text(encoding="utf-8")
old = 'extra_cflags = [opt_level, "-Wno-attributes"]'
new = 'extra_cflags = ["/Od" if FAST_COMPILE else "/O2"]'
if old in txt:
    txt = txt.replace(old, new)
p.write_text(txt, encoding="utf-8")
print("patched", p)
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

Write-Info "Running PyInstaller"
& $py -m PyInstaller --clean --noconfirm packaging\\windows\\sharp-gui.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

Write-Info "Done. Output: dist\\sharp-gui\\sharp-gui.exe"
