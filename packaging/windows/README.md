## Windows GPU self-contained GUI build

This repo includes Windows-only double-click GUI launchers built with PyInstaller:

- `sharp-gui.exe` (predict/render)
- `sharp-usdz-gui.exe` (mesh/USDZ workflows)
- `sharp-splat-viewer-gui.exe` (PLY -> HTML viewer)

### What "self-contained" means

- The build bundles Python + all Python dependencies (including CUDA-enabled PyTorch).
- You still need an NVIDIA driver installed on the target machine.
- The build bundles a prebuilt `gsplat` CUDA extension (`gsplat_cuda.pyd`) so end-users do not need Visual Studio Build Tools.

### Prerequisites (build machine)

- Python 3.12 (`py -3.12`)
- NVIDIA driver + CUDA toolkit (nvcc)
- Visual Studio Build Tools 2022 with "Desktop development with C++"

### Build

From the repo root:

```powershell
.\packaging\windows\build-gpu-gui.ps1
```

Output:

- `dist\sharp-gui\sharp-gui.exe`

For the mesh/USDZ GUI:

```powershell
.\packaging\windows\build-gpu-usdz-gui.ps1
```

Output:

- `dist\sharp-usdz-gui\sharp-usdz-gui.exe`

### Running

Double-click `sharp-gui.exe`. It runs `sharp predict` or `sharp render` in-process and shows output/progress in the GUI.

## Windows HTML viewer GUI (splat-transform)

This repo also includes a Windows-only double-click GUI (`sharp-splat-viewer-gui.exe`) that wraps PlayCanvas
`@playcanvas/splat-transform` to generate a single-page HTML viewer (`.html`) from a `.ply`.

### Build

From the repo root:

```powershell
.\packaging\windows\build-splat-viewer-gui.ps1
```

Output:

- `dist\sharp-splat-viewer-gui\sharp-splat-viewer-gui.exe`
