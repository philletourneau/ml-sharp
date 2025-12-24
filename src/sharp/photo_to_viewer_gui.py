"""Windows GUI for generating a SHARP splat viewer HTML from a photo."""

from __future__ import annotations

import contextlib
import io
import os
import queue
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import click

from sharp.cli import photo_to_viewer


@dataclass(frozen=True)
class RunConfig:
    input_path: str
    output_dir: str | None
    output_html: str | None
    checkpoint_path: str | None
    device: str
    viewer_settings: str | None
    unbundled: bool
    overwrite: bool
    enable_aa: bool
    enable_msaa: bool
    match_gsplat_camera: bool
    splat_gpu: str
    iterations: int
    extra_args: str
    verbose: bool


class _QueueWriter(io.TextIOBase):
    def __init__(self, q: "queue.Queue[str]") -> None:
        self._q = q

    def write(self, s: str) -> int:  # type: ignore[override]
        if s:
            self._q.put(s)
        return len(s)

    def flush(self) -> None:  # noqa: D401
        return


def _build_args(cfg: RunConfig) -> list[str]:
    args: list[str] = ["photo-to-viewer", "-i", cfg.input_path]

    if cfg.output_dir:
        args += ["-o", cfg.output_dir]
    if cfg.output_html:
        args += ["--output-html", cfg.output_html]
    if cfg.checkpoint_path:
        args += ["-c", cfg.checkpoint_path]

    args += ["--device", cfg.device]

    if cfg.viewer_settings:
        args += ["--viewer-settings", cfg.viewer_settings]

    if cfg.unbundled:
        args += ["--unbundled"]

    args += ["--overwrite" if cfg.overwrite else "--no-overwrite"]
    args += ["--aa" if cfg.enable_aa else "--no-aa"]
    args += ["--msaa" if cfg.enable_msaa else "--no-msaa"]
    args += ["--match-gsplat-camera" if cfg.match_gsplat_camera else "--no-match-gsplat-camera"]

    args += ["--splat-gpu", cfg.splat_gpu]
    args += ["--iterations", str(cfg.iterations)]

    if cfg.extra_args.strip():
        args += ["--extra-args", cfg.extra_args.strip()]

    if cfg.verbose:
        args += ["--verbose"]

    return args


def _get_cli_help_text() -> str:
    cmd = photo_to_viewer.photo_to_viewer_cli
    return cmd.get_help(click.Context(cmd))


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("SHARP Photo -> HTML Viewer")
    root.geometry("980x720")

    output_q: queue.Queue[str] = queue.Queue()
    is_running = tk.BooleanVar(value=False)

    def parse_int(s: str, *, default: int) -> int:
        s = (s or "").strip()
        if not s:
            return default
        return int(s)

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
        initial_dir = "Z:/Splats/images"
        initialdir_arg = {"initialdir": initial_dir} if Path(initial_dir).exists() else {}
        p = filedialog.askopenfilename(
            title="Select an input image",
            filetypes=[("Images", "*.*")],
            **initialdir_arg,
        )
        if p:
            input_var.set(p)
            if not output_html_var.get().strip():
                out_dir = output_dir_var.get().strip()
                out_dir = out_dir or str(Path(p).parent)
                output_html_var.set(str(Path(out_dir) / (Path(p).stem + ".html")))

    def browse_output_dir() -> None:
        default_out = output_dir_var.get().strip()
        initialdir_arg = {"initialdir": default_out} if default_out and Path(default_out).exists() else {}
        p = filedialog.askdirectory(title="Select an output folder", **initialdir_arg)
        if p:
            output_dir_var.set(p)
            if input_var.get().strip() and not output_html_var.get().strip():
                output_html_var.set(str(Path(p) / (Path(input_var.get().strip()).stem + ".html")))

    def browse_output_html() -> None:
        default_out = output_html_var.get().strip()
        initial_dir = Path(default_out).parent if default_out else Path(output_dir_var.get().strip() or ".")
        initialdir_arg = {"initialdir": str(initial_dir)} if initial_dir.exists() else {}
        p = filedialog.asksaveasfilename(
            title="Save HTML viewer as...",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("All", "*.*")],
            **initialdir_arg,
        )
        if p:
            output_html_var.set(p)

    def browse_checkpoint() -> None:
        p = filedialog.askopenfilename(
            title="Select a checkpoint (.pt) (optional)",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All", "*.*")],
        )
        if p:
            checkpoint_var.set(p)

    def browse_viewer_settings() -> None:
        p = filedialog.askopenfilename(
            title="Select viewer settings JSON (optional)",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if p:
            viewer_settings_var.set(p)

    def open_output_folder() -> None:
        p = output_dir_var.get().strip()
        if not p and output_html_var.get().strip():
            p = str(Path(output_html_var.get().strip()).parent)
        if not p:
            return
        try:
            os.startfile(p)  # type: ignore[attr-defined]
        except Exception as exc:
            log_line(f"Could not open output folder: {exc}")

    def open_output_html() -> None:
        p = output_html_var.get().strip()
        if not p:
            return
        try:
            os.startfile(p)  # type: ignore[attr-defined]
        except Exception as exc:
            log_line(f"Could not open output HTML: {exc}")

    def make_config() -> RunConfig | None:
        in_path = input_var.get().strip()
        if not in_path:
            log_line("Please select an input image.")
            return None

        out_dir = output_dir_var.get().strip()
        out_html = output_html_var.get().strip()
        if not out_dir and not out_html:
            log_line("Please select an output folder or HTML path.")
            return None

        iterations = parse_int(iterations_var.get(), default=18)

        return RunConfig(
            input_path=in_path,
            output_dir=(out_dir or None),
            output_html=(out_html or None),
            checkpoint_path=(checkpoint_var.get().strip() or None),
            device=device_var.get().strip() or "cuda",
            viewer_settings=(viewer_settings_var.get().strip() or None),
            unbundled=bool(unbundled_var.get()),
            overwrite=bool(overwrite_var.get()),
            enable_aa=bool(enable_aa_var.get()),
            enable_msaa=bool(enable_msaa_var.get()),
            match_gsplat_camera=bool(match_gsplat_camera_var.get()),
            splat_gpu=splat_gpu_var.get().strip() or "default",
            iterations=iterations,
            extra_args=extra_args_var.get(),
            verbose=bool(verbose_var.get()),
        )

    def copy_command() -> None:
        cfg = make_config()
        if cfg is None:
            return
        args = _build_args(cfg)
        cmd = "sharp " + " ".join(shlex.quote(arg) for arg in args)
        root.clipboard_clear()
        root.clipboard_append(cmd)
        log_line("Copied command to clipboard.")

    def show_help() -> None:
        log_line("\n" + _get_cli_help_text() + "\n")

    def run_in_thread(cfg: RunConfig) -> None:
        set_running(True)
        started = time.time()
        try:
            args = _build_args(cfg)
            log_line(f"Running: sharp {' '.join(args)}")

            with contextlib.redirect_stdout(_QueueWriter(output_q)), contextlib.redirect_stderr(
                _QueueWriter(output_q)
            ):
                photo_to_viewer.photo_to_viewer_cli.callback(
                    input_path=Path(cfg.input_path),
                    output_dir=Path(cfg.output_dir) if cfg.output_dir else None,
                    output_html=Path(cfg.output_html) if cfg.output_html else None,
                    checkpoint_path=Path(cfg.checkpoint_path) if cfg.checkpoint_path else None,
                    device=cfg.device,
                    viewer_settings=Path(cfg.viewer_settings) if cfg.viewer_settings else None,
                    unbundled=cfg.unbundled,
                    overwrite=cfg.overwrite,
                    enable_aa=cfg.enable_aa,
                    enable_msaa=cfg.enable_msaa,
                    match_gsplat_camera=cfg.match_gsplat_camera,
                    splat_gpu=cfg.splat_gpu,
                    iterations=cfg.iterations,
                    extra_args=cfg.extra_args,
                    verbose=cfg.verbose,
                )

            log_line(f"Done in {time.time() - started:.1f}s")
        except SystemExit as exc:
            log_line(f"Exited with code {exc.code}")
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

    input_var = tk.StringVar(value="")
    default_output = (
        str(photo_to_viewer.DEFAULT_OUTPUT_DIR)
        if photo_to_viewer.DEFAULT_OUTPUT_DIR.exists()
        else ""
    )
    output_dir_var = tk.StringVar(value=default_output)
    output_html_var = tk.StringVar(value="")
    checkpoint_var = tk.StringVar(value="")
    device_var = tk.StringVar(value="cuda")
    viewer_settings_var = tk.StringVar(value="")
    unbundled_var = tk.IntVar(value=0)
    overwrite_var = tk.IntVar(value=1)
    enable_aa_var = tk.IntVar(value=1)
    enable_msaa_var = tk.IntVar(value=1)
    match_gsplat_camera_var = tk.IntVar(value=1)
    splat_gpu_var = tk.StringVar(value="default")
    iterations_var = tk.StringVar(value="18")
    extra_args_var = tk.StringVar(value="")
    verbose_var = tk.IntVar(value=0)

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    row = 0
    ttk.Label(frm, text="Input image").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=input_var, width=86).grid(row=row, column=1, sticky="we")
    ttk.Button(frm, text="Browse...", command=browse_input).grid(row=row, column=2, padx=(8, 0))

    row += 1
    ttk.Label(frm, text="Output folder").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=output_dir_var, width=86).grid(
        row=row, column=1, sticky="we", pady=(8, 0)
    )
    ttk.Button(frm, text="Browse...", command=browse_output_dir).grid(
        row=row, column=2, padx=(8, 0), pady=(8, 0)
    )

    row += 1
    ttk.Label(frm, text="Output HTML").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=output_html_var, width=86).grid(
        row=row, column=1, sticky="we", pady=(8, 0)
    )
    ttk.Button(frm, text="Save As...", command=browse_output_html).grid(
        row=row, column=2, padx=(8, 0), pady=(8, 0)
    )

    row += 1
    ttk.Label(frm, text="Checkpoint (.pt)").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=checkpoint_var, width=86).grid(
        row=row, column=1, sticky="we", pady=(8, 0)
    )
    ttk.Button(frm, text="Browse...", command=browse_checkpoint).grid(
        row=row, column=2, padx=(8, 0), pady=(8, 0)
    )

    row += 1
    ttk.Label(frm, text="Viewer settings (.json)").grid(row=row, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=viewer_settings_var, width=86).grid(
        row=row, column=1, sticky="we", pady=(8, 0)
    )
    ttk.Button(frm, text="Browse...", command=browse_viewer_settings).grid(
        row=row, column=2, padx=(8, 0), pady=(8, 0)
    )

    row += 1
    opts = ttk.Frame(frm)
    opts.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
    ttk.Checkbutton(opts, text="Overwrite output", variable=overwrite_var).pack(side="left")
    ttk.Checkbutton(opts, text="Unbundled output", variable=unbundled_var).pack(
        side="left", padx=(12, 0)
    )
    ttk.Checkbutton(opts, text="AA", variable=enable_aa_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(opts, text="MSAA", variable=enable_msaa_var).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(
        opts, text="Match gsplat camera", variable=match_gsplat_camera_var
    ).pack(side="left", padx=(12, 0))
    ttk.Checkbutton(opts, text="Verbose", variable=verbose_var).pack(side="left", padx=(12, 0))

    row += 1
    options = ttk.Frame(frm)
    options.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
    ttk.Label(options, text="Device").grid(row=0, column=0, sticky="w")
    device_combo = ttk.Combobox(
        options, textvariable=device_var, values=("cuda", "cpu", "mps", "default"), width=12
    )
    device_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))

    ttk.Label(options, text="Splat GPU").grid(row=0, column=2, sticky="w", padx=(16, 0))
    ttk.Entry(options, textvariable=splat_gpu_var, width=12).grid(
        row=0, column=3, sticky="w", padx=(8, 0)
    )

    ttk.Label(options, text="Iterations").grid(row=0, column=4, sticky="w", padx=(16, 0))
    ttk.Entry(options, textvariable=iterations_var, width=8).grid(
        row=0, column=5, sticky="w", padx=(8, 0)
    )

    row += 1
    ttk.Label(frm, text="Extra splat-transform args").grid(
        row=row, column=0, sticky="w", pady=(8, 0)
    )
    ttk.Entry(frm, textvariable=extra_args_var, width=86).grid(
        row=row, column=1, sticky="we", pady=(8, 0)
    )

    row += 1
    btns = ttk.Frame(frm)
    btns.grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 0))
    run_btn = ttk.Button(btns, text="Run", command=on_run)
    run_btn.pack(side="left")
    copy_btn = ttk.Button(btns, text="Copy Command", command=copy_command)
    copy_btn.pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="Help", command=show_help).pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="Open Output Folder", command=open_output_folder).pack(
        side="left", padx=(8, 0)
    )
    ttk.Button(btns, text="Open Output HTML", command=open_output_html).pack(
        side="left", padx=(8, 0)
    )
    ttk.Button(btns, text="Clear Log", command=on_clear_log).pack(side="left", padx=(8, 0))

    row += 1
    log_text = tk.Text(frm, height=12, wrap="word", state="disabled")
    log_text.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(12, 0))

    frm.columnconfigure(1, weight=1)
    frm.rowconfigure(row, weight=1)

    drain_output()
    root.mainloop()
