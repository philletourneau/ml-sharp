$ErrorActionPreference = "Stop"

function Write-Info($msg) { Write-Host "[build] $msg" }

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path

$repoRoot = (Get-Location).Path
$venv = Join-Path $repoRoot ".venv-build-splat-viewer-gui"
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

Write-Info "Creating build venv at $venv"
py -3.12 -m venv $venv
$py = Join-Path $venv "Scripts\\python.exe"
& $py -m pip install --upgrade pip setuptools wheel
& $py -m pip install pyinstaller

Write-Info "Running PyInstaller"
& $py -m PyInstaller --clean --noconfirm packaging\\windows\\sharp-splat-viewer-gui.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

Write-Info "Done. Output: dist\\sharp-splat-viewer-gui\\sharp-splat-viewer-gui.exe"
