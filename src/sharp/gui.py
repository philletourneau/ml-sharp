"""Windows GUI launcher for SHARP.

This provides a double-clickable entry point that prompts for input/output paths,
exposes common flags, and runs the existing click-based CLI in-process.
"""

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

from sharp.cli import main_cli


@dataclass(frozen=True)
class RunConfig:
    mode: str  # "predict" or "render"
    input_path: str
    output_path: str
    checkpoint_path: str | None
    fps: float
    duration_scale: float
    trajectory_variants_count: int
    trajectory_variants: bool
    output_prefix: str
    progress: bool
    verbose: bool
    extra_args: str


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
    args: list[str] = [cfg.mode, "-i", cfg.input_path, "-o", cfg.output_path]

    if cfg.mode == "predict" and cfg.checkpoint_path:
        args += ["-c", cfg.checkpoint_path]

    if cfg.mode == "predict":
        args += ["--render", "--device", "cuda"]

    args += ["--fps", str(cfg.fps), "--duration-scale", str(cfg.duration_scale)]

    if cfg.output_prefix:
        args += ["--output-prefix", cfg.output_prefix]

    if cfg.trajectory_variants:
        args += ["--trajectory-variants"]
        args += ["--trajectory-variants-count", str(cfg.trajectory_variants_count)]

    if not cfg.progress:
        args += ["--no-progress"]

    if cfg.verbose:
        args += ["--verbose"]

    if cfg.extra_args.strip():
        args += shlex.split(cfg.extra_args)

    return args


def _get_cli_help_text(command: str) -> str:
    ctx = click.Context(main_cli)
    if command == "predict":
        cmd = main_cli.get_command(ctx, "predict")
    else:
        cmd = main_cli.get_command(ctx, "render")
    if cmd is None:
        return "Command not found."
    return cmd.get_help(click.Context(cmd))


def main() -> None:
    # Tkinter is Windows built-in; import lazily so `sharp` CLI stays lightweight.
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("SHARP (GPU) Launcher")
    root.geometry("980x680")

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

    def browse_input() -> None:
        mode = mode_var.get()
        initial_dir = "Z:/Splats/images"
        if Path(initial_dir).exists():
            initialdir_arg = {"initialdir": initial_dir}
        else:
            initialdir_arg = {}
        if mode == "predict":
            p = filedialog.askopenfilename(
                title="Select an input image",
                filetypes=[("Images", "*.*")],
                **initialdir_arg,
            )
        else:
            p = filedialog.askopenfilename(
                title="Select a .ply file",
                filetypes=[("PLY", "*.ply"), ("All", "*.*")],
                **initialdir_arg,
            )
        if p:
            input_var.set(p)

    def browse_output() -> None:
        default_out = output_var.get().strip() or "Z:/Splats/output"
        initialdir_arg = {"initialdir": default_out} if Path(default_out).exists() else {}
        p = filedialog.askdirectory(title="Select an output folder", **initialdir_arg)
        if p:
            output_var.set(p)

    def browse_checkpoint() -> None:
        p = filedialog.askopenfilename(
            title="Select a checkpoint (.pt) (optional)",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All", "*.*")],
        )
        if p:
            checkpoint_var.set(p)

    def show_help() -> None:
        txt = _get_cli_help_text(mode_var.get())
        log_line("\n" + txt + "\n")

    def run_in_thread(cfg: RunConfig) -> None:
        set_running(True)
        started = time.time()
        try:
            args = _build_args(cfg)
            log_line(f"Running: sharp {' '.join(args)}")

            # For GUI output, prefer the plain progress output over tqdm bars.
            env = os.environ.copy()
            env.setdefault("SHARP_PROGRESS_STYLE", "text")
            os.environ.update(env)

            try:
                with contextlib.redirect_stdout(_QueueWriter(output_q)), contextlib.redirect_stderr(
                    _QueueWriter(output_q)
                ):
                    main_cli.main(args=args, standalone_mode=False)
            except Exception as exc:
                msg = str(exc)
                if "Failed to download the default checkpoint" in msg or "CERTIFICATE_VERIFY_FAILED" in msg:
                    log_line(
                        "Download failed. If you're behind a corporate proxy/SSL interceptor, install your "
                        "organization's root certificate into the Windows trust store, or manually download "
                        "the checkpoint and select it via the 'Checkpoint (.pt)' field."
                    )
                    log_line(
                        "You can also set `SHARP_DISABLE_TRUSTSTORE=1` to disable using the Windows trust store."
                    )
                raise
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

        in_path = input_var.get().strip()
        out_path = output_var.get().strip()
        if not in_path or not out_path:
            log_line("Please select both input and output paths.")
            return

        ckpt = checkpoint_var.get().strip() or None
        variants_count = int(variants_count_var.get() or "1")
        cfg = RunConfig(
            mode=mode_var.get(),
            input_path=in_path,
            output_path=out_path,
            checkpoint_path=ckpt,
            fps=float(fps_var.get() or "60"),
            duration_scale=float(duration_var.get() or "8"),
            trajectory_variants_count=variants_count,
            trajectory_variants=(variants_count > 1),
            output_prefix="",
            progress=bool(progress_var.get()),
            verbose=bool(verbose_var.get()),
            extra_args=extra_var.get(),
        )

        t = threading.Thread(target=run_in_thread, args=(cfg,), daemon=True)
        t.start()

    def on_previz() -> None:
        if is_running.get():
            return

        in_path = input_var.get().strip()
        out_path = output_var.get().strip()
        if not in_path or not out_path:
            log_line("Please select both input and output paths.")
            return

        ckpt = checkpoint_var.get().strip() or None
        previz_fps = 30.0
        previz_duration_s = 1.0
        base_steps = 60
        try:
            from sharp.utils import camera

            base_steps = camera.TrajectoryParams().num_steps
        except Exception:
            pass

        # duration_scale multiplies the default number of steps; choose it to hit the desired duration.
        duration_scale = max(1.0 / base_steps, (previz_fps * previz_duration_s) / float(base_steps))

        cfg = RunConfig(
            mode=mode_var.get(),
            input_path=in_path,
            output_path=out_path,
            checkpoint_path=ckpt,
            fps=previz_fps,
            duration_scale=duration_scale,
            trajectory_variants_count=1,
            trajectory_variants=False,
            output_prefix="previz_",
            progress=True,
            verbose=bool(verbose_var.get()),
            extra_args=extra_var.get(),
        )

        t = threading.Thread(target=run_in_thread, args=(cfg,), daemon=True)
        t.start()

    # Top controls
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill="both", expand=True)

    mode_var = tk.StringVar(value="predict")
    input_var = tk.StringVar()
    output_var = tk.StringVar(value="Z:/Splats/output")
    checkpoint_var = tk.StringVar()
    fps_var = tk.StringVar(value="60")
    duration_var = tk.StringVar(value="8")
    variants_count_var = tk.StringVar(value="5")
    progress_var = tk.IntVar(value=1)
    verbose_var = tk.IntVar(value=0)
    extra_var = tk.StringVar(value="")

    row = 0
    ttk.Label(frm, text="Mode").grid(row=row, column=0, sticky="w")
    ttk.Radiobutton(frm, text="Predict + Render (GPU)", value="predict", variable=mode_var).grid(
        row=row, column=1, sticky="w"
    )
    ttk.Radiobutton(frm, text="Render from PLY (GPU)", value="render", variable=mode_var).grid(
        row=row, column=2, sticky="w"
    )

    row += 1
    ttk.Label(frm, text="Input").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=input_var, width=90).grid(row=row, column=1, columnspan=2, sticky="we")
    ttk.Button(frm, text="Browse…", command=browse_input).grid(row=row, column=3, sticky="e")

    row += 1
    ttk.Label(frm, text="Output folder").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=output_var, width=90).grid(row=row, column=1, columnspan=2, sticky="we")
    ttk.Button(frm, text="Browse…", command=browse_output).grid(row=row, column=3, sticky="e")

    row += 1
    ttk.Label(frm, text="Checkpoint (.pt) (optional)").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=checkpoint_var, width=90).grid(
        row=row, column=1, columnspan=2, sticky="we"
    )
    ttk.Button(frm, text="Browse…", command=browse_checkpoint).grid(row=row, column=3, sticky="e")

    row += 1
    ttk.Label(frm, text="FPS").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=fps_var, width=10).grid(row=row, column=1, sticky="w")
    ttk.Label(frm, text="Duration scale").grid(row=row, column=2, sticky="w")
    ttk.Entry(frm, textvariable=duration_var, width=10).grid(row=row, column=3, sticky="w")

    row += 1
    ttk.Label(frm, text="Trajectory variants").grid(row=row, column=0, sticky="w")
    variants_combo = ttk.Combobox(
        frm, textvariable=variants_count_var, values=("1", "2", "3", "4", "5"), width=8, state="readonly"
    )
    variants_combo.grid(row=row, column=1, sticky="w")
    ttk.Checkbutton(frm, text="Progress", variable=progress_var).grid(row=row, column=2, sticky="w")
    ttk.Checkbutton(frm, text="Verbose", variable=verbose_var).grid(row=row, column=3, sticky="w")

    row += 1
    ttk.Label(frm, text="Extra args").grid(row=row, column=0, sticky="w")
    ttk.Entry(frm, textvariable=extra_var, width=90).grid(row=row, column=1, columnspan=3, sticky="we")

    row += 1
    btns = ttk.Frame(frm)
    btns.grid(row=row, column=0, columnspan=4, sticky="we", pady=(8, 8))
    run_btn = ttk.Button(btns, text="Run", command=on_run)
    run_btn.pack(side="left")
    ttk.Button(btns, text="Previz (fast)", command=on_previz).pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="Show Flags/Help", command=show_help).pack(side="left", padx=(8, 0))
    ttk.Button(btns, text="Clear Log", command=lambda: (log_text.configure(state="normal"), log_text.delete("1.0", "end"), log_text.configure(state="disabled"))).pack(
        side="left", padx=(8, 0)
    )

    # Log output
    row += 1
    ttk.Label(frm, text="Output").grid(row=row, column=0, sticky="w")
    row += 1
    log_text = tk.Text(frm, height=22, wrap="word", state="disabled")
    log_text.grid(row=row, column=0, columnspan=4, sticky="nsew")
    scrollbar = ttk.Scrollbar(frm, command=log_text.yview)
    scrollbar.grid(row=row, column=4, sticky="ns")
    log_text.configure(yscrollcommand=scrollbar.set)

    frm.columnconfigure(1, weight=1)
    frm.columnconfigure(2, weight=1)
    frm.rowconfigure(row, weight=1)

    drain_output()
    log_line("Tip: Click 'Show Flags/Help' to see all CLI options.")
    root.mainloop()
