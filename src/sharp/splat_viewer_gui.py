"""Windows GUI for generating a PlayCanvas HTML splat viewer using splat-transform.

This app wraps the `@playcanvas/splat-transform` CLI and produces a single-page HTML viewer.
"""

from __future__ import annotations

import os
import queue
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    input_path: Path
    output_html: Path
    viewer_settings: Path | None
    unbundled: bool
    overwrite: bool
    gpu: str | None  # "cpu" or an integer string
    iterations: int | None
    extra_args: str


def _apply_sharp_style_viewer_constraints(output_html: Path, *, unbundled: bool) -> None:
    """Post-process SuperSplat Viewer to restrict camera controls (SHARP-style)."""

    def patch_text_file(path: Path, patcher: Callable[[str], str]) -> None:
        text = path.read_text(encoding="utf-8")
        patched = patcher(text)
        if patched != text:
            path.write_text(patched, encoding="utf-8")

    # Hide fly camera button in the UI (if present).
    if output_html.is_file():
        def hide_fly_button(text: str) -> str:
            if 'id="flyCamera"' not in text:
                return text
            if 'id="flyCamera" style="display:none"' in text:
                return text
            if '<button id="flyCamera"' in text:
                return text.replace(
                    '<button id="flyCamera"',
                    '<button id="flyCamera" style="display:none"',
                    1,
                )
            return text

        patch_text_file(output_html, hide_fly_button)

    js_path = output_html if not unbundled else output_html.parent / "index.js"
    if not js_path.is_file():
        raise RuntimeError(f"Expected viewer script not found: {js_path}")

    def patch_viewer_js(text: str) -> str:
        patched = text

        # Disable fly mode via keyboard auto-switch.
        no_fly_auto_switch = (
            "if (state.cameraMode !== 'fly' && this._state.axis.length() > 0) {\n"
            "            state.cameraMode = 'fly';\n"
            "        }"
        )
        patched = patched.replace(no_fly_auto_switch, "")

        # Disable panning (right-mouse / 2-finger) so the target stays fixed.
        if "const pan = 0;" not in patched:
            if "const pan = this._state.mouse[2] || +(button[2] === -1) || double;" not in patched:
                raise RuntimeError("Could not disable panning (expected pan expression not found).")
            patched = patched.replace(
                "const pan = this._state.mouse[2] || +(button[2] === -1) || double;",
                "const pan = 0;",
                1,
            )

        # Clamp fly -> orbit if anything tries to switch modes.
        if "property === 'cameraMode' && value === 'fly'" not in patched:
            if "// not allowed to set a new value on target" not in patched:
                raise RuntimeError("Could not patch cameraMode (anchor comment not found).")
            patched = patched.replace(
                "// not allowed to set a new value on target",
                "if (property === 'cameraMode' && value === 'fly') {\n"
                "                value = 'orbit';\n"
                "            }\n"
                "            // not allowed to set a new value on target",
                1,
            )

        # Limit orbit pitch + zoom around the current pose (approx SHARP turntable feel).
        orbit_limits_marker = "this.controller.zoomRange = new Vec2(dist * 0.85, dist * 1.15);"
        if orbit_limits_marker not in patched:
            if "this.controller.attach(p, false);" not in patched:
                raise RuntimeError("Could not patch orbit camera limits (attach(p,false) not found).")
            patched = patched.replace(
                "this.controller.attach(p, false);",
                "this.controller.yawRange = new Vec2(-180, 180);\n"
                "        const pitch = camera.angles.x;\n"
                "        this.controller.pitchRange = new Vec2(pitch - 20, pitch + 20);\n"
                "        const dist = Math.max(camera.distance, 1e-6);\n"
                f"        {orbit_limits_marker}\n"
                "        this.controller.attach(p, false);",
                1,
            )
            if "this.controller.attach(p, true);" not in patched:
                raise RuntimeError("Could not patch orbit camera limits (attach(p,true) not found).")
            patched = patched.replace(
                "this.controller.attach(p, true);",
                "this.controller.yawRange = new Vec2(-180, 180);\n"
                "        const pitch = camera.angles.x;\n"
                "        this.controller.pitchRange = new Vec2(pitch - 20, pitch + 20);\n"
                "        const dist = Math.max(camera.distance, 1e-6);\n"
                f"        {orbit_limits_marker}\n"
                "        this.controller.attach(p, true);",
                1,
            )

        return patched

    patch_text_file(js_path, patch_viewer_js)


_PLY_SCALAR_TYPE_SIZES: dict[str, int] = {
    "char": 1,
    "int8": 1,
    "uchar": 1,
    "uint8": 1,
    "short": 2,
    "int16": 2,
    "ushort": 2,
    "uint16": 2,
    "int": 4,
    "int32": 4,
    "uint": 4,
    "uint32": 4,
    "float": 4,
    "float32": 4,
    "double": 8,
    "float64": 8,
}


def _parse_ply_header(path: Path) -> tuple[str, list[tuple[str, int, list[str]]], int]:
    """Parse enough of a PLY header to locate element boundaries.

    Returns:
        format_line: The `format ...` header line (without newline).
        elements: List of (name, count, property_lines).
        data_start: Byte offset where element data starts (after end_header newline).
    """
    with path.open("rb") as f:
        first = f.readline()
        if not first:
            raise RuntimeError("Empty file.")
        if first.strip() != b"ply":
            raise RuntimeError("Not a PLY file (missing 'ply' header).")

        format_line_b = f.readline()
        if not format_line_b:
            raise RuntimeError("Truncated PLY header (missing format line).")
        format_line = format_line_b.decode("ascii", errors="strict").strip()
        if not format_line.startswith("format "):
            raise RuntimeError("Invalid PLY header (missing format line).")

        elements: list[tuple[str, int, list[str]]] = []
        current_name: str | None = None
        current_count: int | None = None
        current_props: list[str] = []

        while True:
            line_b = f.readline()
            if not line_b:
                raise RuntimeError("Truncated PLY header (missing end_header).")
            line = line_b.decode("ascii", errors="strict").strip()
            if not line or line.startswith("comment"):
                continue
            if line == "end_header":
                if current_name is not None and current_count is not None:
                    elements.append((current_name, current_count, current_props))
                data_start = f.tell()
                return format_line, elements, data_start

            parts = line.split()
            if parts[0] == "element":
                if len(parts) != 3:
                    raise RuntimeError(f"Invalid element line: {line}")
                if current_name is not None and current_count is not None:
                    elements.append((current_name, current_count, current_props))
                current_name = parts[1]
                current_count = int(parts[2])
                current_props = []
            elif parts[0] == "property":
                if current_name is None:
                    continue
                current_props.append(line)


def _element_fixed_record_size(prop_lines: list[str]) -> int:
    size = 0
    for line in prop_lines:
        parts = line.split()
        if len(parts) < 3 or parts[0] != "property":
            continue
        if parts[1] == "list":
            raise RuntimeError(
                "PLY contains variable-length list properties; cannot strip metadata safely."
            )
        prop_type = parts[1].lower()
        if prop_type not in _PLY_SCALAR_TYPE_SIZES:
            raise RuntimeError(f"Unsupported PLY scalar type: {prop_type}")
        size += _PLY_SCALAR_TYPE_SIZES[prop_type]
    return size


def _needs_vertex_only_ply(path: Path) -> bool:
    if path.suffix.lower() != ".ply":
        return False
    format_line, elements, _data_start = _parse_ply_header(path)
    if not format_line.startswith("format binary_"):
        # splat-transform supports ASCII PLY too, but SHARP emits binary; keep this simple.
        return False
    return not (len(elements) == 1 and elements[0][0] == "vertex")


def _write_vertex_only_ply(src: Path, dst: Path) -> None:
    format_line, elements, data_start = _parse_ply_header(src)
    if not format_line.startswith("format binary_"):
        raise RuntimeError(f"Unsupported PLY format for stripping: {format_line}")

    element_names = [name for name, _count, _props in elements]
    if "vertex" not in element_names:
        raise RuntimeError("PLY does not contain a 'vertex' element.")
    vertex_index = element_names.index("vertex")

    sizes = []
    for name, count, props in elements:
        record_size = _element_fixed_record_size(props)
        sizes.append((name, count, record_size, count * record_size))

    vertex_name, vertex_count, vertex_props = elements[vertex_index]
    vertex_total_size = sizes[vertex_index][3]
    vertex_offset = data_start + sum(total for _n, _c, _r, total in sizes[:vertex_index])

    header_lines = [
        "ply",
        format_line,
        "comment generated by sharp-splat-viewer-gui (vertex-only copy for splat-transform)",
        f"element {vertex_name} {vertex_count}",
        *vertex_props,
        "end_header",
    ]
    header = ("\n".join(header_lines) + "\n").encode("ascii")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as f_in, dst.open("wb") as f_out:
        f_out.write(header)
        f_in.seek(vertex_offset)
        remaining = vertex_total_size
        while remaining:
            chunk = f_in.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("Unexpected EOF while copying vertex data.")
            f_out.write(chunk)
            remaining -= len(chunk)


def _find_node_and_cli() -> tuple[Path, Path, Path]:
    """Return (node_exe, cli_mjs, vendor_root)."""
    candidate_roots: list[Path] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate_roots.append(Path(meipass))

    exe_dir = Path(sys.executable).resolve().parent
    candidate_roots += [exe_dir, exe_dir / "_internal"]

    for root in dict.fromkeys(candidate_roots):
        node_exe = root / "node" / "node.exe"
        vendor_root = root / "splat-transform"
        cli_mjs = (
            vendor_root
            / "node_modules"
            / "@playcanvas"
            / "splat-transform"
            / "bin"
            / "cli.mjs"
        )
        if node_exe.is_file() and cli_mjs.is_file():
            return node_exe, cli_mjs, vendor_root

    # Fall back to system node + global install if present.
    node_on_path = shutil.which("node.exe") or shutil.which("node")
    splat_transform_on_path = shutil.which("splat-transform.cmd") or shutil.which("splat-transform")
    if node_on_path and splat_transform_on_path:
        # Best-effort: assume `splat-transform` is runnable directly.
        return Path(node_on_path), Path(splat_transform_on_path), Path.cwd()

    raise RuntimeError(
        "Could not find bundled Node/splat-transform. Rebuild the app, or install Node and "
        "`npm install -g @playcanvas/splat-transform` and try again."
    )


def _build_command(cfg: RunConfig) -> tuple[list[str], Path]:
    node_exe, cli, vendor_root = _find_node_and_cli()

    # If cli is actually `splat-transform.cmd`, we can invoke it directly without node.
    if cli.name.lower().endswith(".cmd"):
        base = [str(cli)]
    else:
        base = [str(node_exe), str(cli)]

    args: list[str] = []
    if cfg.overwrite:
        args += ["-w"]
    if cfg.viewer_settings is not None:
        args += ["-E", str(cfg.viewer_settings)]
    if cfg.unbundled:
        args += ["-U"]
    if cfg.gpu:
        args += ["-g", cfg.gpu]
    if cfg.iterations is not None:
        args += ["-i", str(int(cfg.iterations))]

    args += [str(cfg.input_path), str(cfg.output_html)]

    if cfg.extra_args.strip():
        args += shlex.split(cfg.extra_args, posix=os.name != "nt")

    return base + args, vendor_root


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("SHARP -> PlayCanvas HTML Viewer (splat-transform)")
    root.geometry("980x700")

    output_q: queue.Queue[str] = queue.Queue()
    is_running = tk.BooleanVar(value=False)

    def log_line(s: str) -> None:
        output_q.put(s + ("\n" if not s.endswith("\n") else ""))

    def drain_output() -> None:
        try:
            while True:
                chunk = output_q.get_nowait()
                log_text.configure(state="normal")
                log_text.insert("end", chunk)
                log_text.see("end")
                log_text.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(100, drain_output)

    def set_running(running: bool) -> None:
        is_running.set(running)
        run_btn.configure(state=("disabled" if running else "normal"))
        copy_btn.configure(state=("disabled" if running else "normal"))

    def browse_input() -> None:
        initial_dir = "Z:/Splats/output"
        initialdir_arg = {"initialdir": initial_dir} if Path(initial_dir).exists() else {}
        p = filedialog.askopenfilename(
            title="Select a .ply file",
            filetypes=[("PLY", "*.ply"), ("All", "*.*")],
            **initialdir_arg,
        )
        if p:
            input_var.set(p)
            if not output_var.get().strip():
                out_dir = Path(output_folder_var.get().strip() or "Z:/Splats/output")
                output_var.set(str(out_dir / (Path(p).stem + ".html")))

    def browse_output_html() -> None:
        default_out = output_var.get().strip()
        initial_dir = Path(default_out).parent if default_out else Path(output_folder_var.get())
        initialdir_arg = {"initialdir": str(initial_dir)} if initial_dir.exists() else {}
        p = filedialog.asksaveasfilename(
            title="Save HTML viewer as...",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("All", "*.*")],
            **initialdir_arg,
        )
        if p:
            output_var.set(p)

    def browse_viewer_settings() -> None:
        p = filedialog.askopenfilename(
            title="Select viewer settings JSON (optional)",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if p:
            settings_var.set(p)

    def open_output_html() -> None:
        p = output_var.get().strip()
        if not p:
            return
        try:
            os.startfile(p)  # type: ignore[attr-defined]
        except Exception as exc:
            log_line(f"Could not open output: {exc}")

    def make_config() -> RunConfig | None:
        in_path = input_var.get().strip()
        out_html = output_var.get().strip()
        if not in_path or not out_html:
            log_line("Please select both input and output paths.")
            return None

        settings = settings_var.get().strip()
        settings_path = Path(settings) if settings else None
        gpu = gpu_var.get().strip()
        gpu_value = None
        if gpu and gpu.lower() != "default":
            gpu_value = gpu

        iters = iterations_var.get().strip()
        iterations = int(iters) if iters else None

        return RunConfig(
            input_path=Path(in_path),
            output_html=Path(out_html),
            viewer_settings=settings_path,
            unbundled=bool(unbundled_var.get()),
            overwrite=bool(overwrite_var.get()),
            gpu=gpu_value,
            iterations=iterations,
            extra_args=extra_args_var.get(),
        )

    def copy_command() -> None:
        cfg = make_config()
        if cfg is None:
            return
        try:
            cmd, _cwd = _build_command(cfg)
        except Exception as exc:
            log_line(f"ERROR: {exc}")
            return
        root.clipboard_clear()
        root.clipboard_append(" ".join(cmd))
        log_line("Copied command to clipboard.")

    def show_help() -> None:
        try:
            node_exe, cli, vendor_root = _find_node_and_cli()
            if cli.name.lower().endswith(".cmd"):
                cmd = [str(cli), "--help"]
                cwd = None
            else:
                cmd = [str(node_exe), str(cli), "--help"]
                cwd = vendor_root
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            log_line("\n" + (proc.stdout or proc.stderr or "(no output)") + "\n")
        except Exception as exc:
            log_line(f"ERROR: {exc}")

    def run_in_thread(cfg: RunConfig) -> None:
        set_running(True)
        started = time.time()
        try:
            effective_cfg = cfg
            temp_dir: tempfile.TemporaryDirectory[str] | None = None
            if _needs_vertex_only_ply(cfg.input_path):
                log_line(
                    "Input PLY contains extra metadata elements; generating a vertex-only copy for "
                    "splat-transform..."
                )
                temp_dir = tempfile.TemporaryDirectory(prefix="sharp-splat-transform-")
                stripped = Path(temp_dir.name) / f"{cfg.input_path.stem}.vertex-only.ply"
                _write_vertex_only_ply(cfg.input_path, stripped)
                effective_cfg = RunConfig(
                    input_path=stripped,
                    output_html=cfg.output_html,
                    viewer_settings=cfg.viewer_settings,
                    unbundled=cfg.unbundled,
                    overwrite=cfg.overwrite,
                    gpu=cfg.gpu,
                    iterations=cfg.iterations,
                    extra_args=cfg.extra_args,
                )

            cmd, cwd = _build_command(effective_cfg)
            log_line("Running: " + " ".join(cmd))

            effective_cfg.output_html.parent.mkdir(parents=True, exist_ok=True)

            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                output_q.put(line)
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"splat-transform exited with code {rc}")

            if bool(limit_camera_var.get()):
                log_line("Applying SHARP-style camera limits to the generated viewer...")
                _apply_sharp_style_viewer_constraints(
                    effective_cfg.output_html, unbundled=effective_cfg.unbundled
                )

            log_line(f"Done in {time.time() - started:.1f}s")
        except Exception as exc:
            log_line(f"ERROR: {exc}")
        finally:
            try:
                if temp_dir is not None:
                    temp_dir.cleanup()
            except Exception:
                pass
            set_running(False)

    def on_run() -> None:
        if is_running.get():
            return
        cfg = make_config()
        if cfg is None:
            return
        threading.Thread(target=run_in_thread, args=(cfg,), daemon=True).start()

    def on_clear_log() -> None:
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")

    # Vars
    input_var = tk.StringVar(value="")
    output_folder_var = tk.StringVar(value="Z:/Splats/output")
    output_var = tk.StringVar(value="")
    settings_var = tk.StringVar(value="")
    unbundled_var = tk.IntVar(value=0)
    overwrite_var = tk.IntVar(value=1)
    limit_camera_var = tk.IntVar(value=1)
    gpu_var = tk.StringVar(value="default")
    iterations_var = tk.StringVar(value="")
    extra_args_var = tk.StringVar(value="")

    # Layout
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    row = 0
    ttk.Label(frm, text="Input (.ply)").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=input_var, width=86).grid(row=row, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_input).grid(row=row, column=2, padx=(8, 0))

    row += 1
    ttk.Label(frm, text="Output (.html)").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=output_var, width=86).grid(
        row=row, column=1, sticky="we", pady=(8, 0)
    )
    ttk.Button(frm, text="Save As...", command=browse_output_html).grid(
        row=row, column=2, padx=(8, 0), pady=(8, 0)
    )

    row += 1
    ttk.Label(frm, text="Viewer settings (.json)").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=settings_var, width=86).grid(
        row=row, column=1, sticky="we", pady=(8, 0)
    )
    ttk.Button(frm, text="Browse...", command=browse_viewer_settings).grid(
        row=row, column=2, padx=(8, 0), pady=(8, 0)
    )

    row += 1
    opts = ttk.Frame(frm)
    opts.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
    ttk.Checkbutton(opts, text="Overwrite output", variable=overwrite_var).pack(side="left")
    ttk.Checkbutton(opts, text="Unbundled output (-U)", variable=unbundled_var).pack(
        side="left", padx=(12, 0)
    )
    ttk.Checkbutton(opts, text="Limit camera controls (SHARP-style)", variable=limit_camera_var).pack(
        side="left", padx=(12, 0)
    )
    ttk.Label(opts, text="GPU").pack(side="left", padx=(12, 0))
    ttk.Entry(opts, textvariable=gpu_var, width=10).pack(side="left")
    ttk.Label(opts, text="Iterations").pack(side="left", padx=(12, 0))
    ttk.Entry(opts, textvariable=iterations_var, width=8).pack(side="left")

    row += 1
    ttk.Label(frm, text="Extra args").grid(row=row, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(frm, textvariable=extra_args_var, width=86).grid(
        row=row, column=1, columnspan=2, sticky="we", pady=(10, 0)
    )

    row += 1
    btns = ttk.Frame(frm)
    btns.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 10))
    run_btn = ttk.Button(btns, text="Generate Viewer", command=on_run)
    run_btn.pack(side="left")
    ttk.Button(btns, text="Open Output", command=open_output_html).pack(side="left", padx=(8, 0))
    copy_btn = ttk.Button(btns, text="Copy Command", command=copy_command)
    copy_btn.pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="Show splat-transform Help", command=show_help).pack(
        side="left", padx=(8, 0)
    )
    ttk.Button(btns, text="Clear Log", command=on_clear_log).pack(side="left", padx=(8, 0))

    row += 1
    ttk.Label(frm, text="Output").grid(row=row, column=0, sticky="w")
    row += 1
    log_text = tk.Text(frm, height=24, wrap="word", state="disabled")
    log_text.grid(row=row, column=0, columnspan=3, sticky="nsew")
    scrollbar = ttk.Scrollbar(frm, command=log_text.yview)
    scrollbar.grid(row=row, column=3, sticky="ns")
    log_text.configure(yscrollcommand=scrollbar.set)

    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(row, weight=1)

    drain_output()
    log_line("Tip: Defaults to a bundled single-page HTML viewer. Use -U for unbundled output.")
    root.mainloop()
