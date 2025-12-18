# PyInstaller spec for a Windows-only self-contained splat-transform HTML viewer GUI build.
#
# Expected build input:
# - Node extracted to `packaging/windows/assets/node`
# - `@playcanvas/splat-transform` installed to `packaging/windows/assets/splat-transform`
#
# Build:
#   pyinstaller --clean --noconfirm packaging/windows/sharp-splat-viewer-gui.spec

from pathlib import Path
import os

from PyInstaller.building.datastruct import Tree

# PyInstaller executes spec files via `exec` and does not guarantee `__file__`.
# We assume the build is invoked from the repo root.
ROOT = Path(os.getcwd()).resolve()
ASSETS = ROOT / "packaging" / "windows" / "assets"

block_cipher = None

node_tree = Tree(str(ASSETS / "node"), prefix="node") if (ASSETS / "node").is_dir() else None
vendor_tree = (
    Tree(str(ASSETS / "splat-transform"), prefix="splat-transform")
    if (ASSETS / "splat-transform").is_dir()
    else None
)

a = Analysis(
    [str(ROOT / "packaging" / "windows" / "sharp_splat_viewer_gui_entry.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

if node_tree is not None:
    a.datas += node_tree
if vendor_tree is not None:
    a.datas += vendor_tree

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sharp-splat-viewer-gui",
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
    name="sharp-splat-viewer-gui",
)
