# PyInstaller spec for a Windows-only self-contained GPU GUI build.
#
# Expected build input:
# - a built gsplat CUDA extension copied to `packaging/windows/assets/gsplat_cuda.pyd`
#
# Build:
#   pyinstaller --clean --noconfirm packaging/windows/sharp-gui.spec

from pathlib import Path
import os

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# PyInstaller executes spec files via `exec` and does not guarantee `__file__`.
# We assume the build is invoked from the repo root.
ROOT = Path(os.getcwd()).resolve()
ASSETS = ROOT / "packaging" / "windows" / "assets"

block_cipher = None

hiddenimports = (
    collect_submodules("gsplat")
    + collect_submodules("imageio")
    + collect_submodules("imageio_ffmpeg")
    + collect_submodules("numpy")
    + collect_submodules("torch")
    + collect_submodules("torchvision")
)

datas = []
# Some packages (e.g. imageio) call `importlib.metadata.version()` at import time.
# Ensure package metadata is bundled so those lookups work in the frozen app.
datas += copy_metadata("imageio")
datas += copy_metadata("imageio-ffmpeg")
datas += copy_metadata("gsplat")
datas += copy_metadata("sharp")
datas += copy_metadata("truststore")
gsplat_ext = ASSETS / "gsplat_cuda.pyd"
if gsplat_ext.exists():
    datas.append((str(gsplat_ext), "."))

a = Analysis(
    [str(ROOT / "packaging" / "windows" / "sharp_gui_entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sharp-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="sharp-gui",
)
