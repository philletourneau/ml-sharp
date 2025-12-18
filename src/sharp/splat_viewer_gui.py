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
import threading
import time
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
            cmd, cwd = _build_command(cfg)
            log_line("Running: " + " ".join(cmd))

            cfg.output_html.parent.mkdir(parents=True, exist_ok=True)

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
            log_line(f"Done in {time.time() - started:.1f}s")
        except Exception as exc:
            log_line(f"ERROR: {exc}")
        finally:
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
