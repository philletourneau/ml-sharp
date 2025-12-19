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
    device: str
    with_rendering: bool

    fps: float
    write_frames: bool
    frame_ext: str
    jpeg_quality: int
    video_ext: str
    codec: str | None
    bitrate: str | None
    macro_block_size: int
    render_depth: bool
    duration_scale: float
    trajectory_type: str | None
    lookat_mode: str | None
    max_disparity: float | None
    max_zoom: float | None
    distance_m: float | None
    num_steps: int | None
    num_repeats: int | None
    trajectory_variants: bool
    trajectory_variants_count: int
    output_prefix: str
    overwrite: bool
    low_pass_filter_eps: float
    cuda_device: int

    progress: bool
    progress_style: str  # "auto" or "text"
    verbose: bool
    disable_truststore: bool
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

    if cfg.mode == "predict":
        if cfg.checkpoint_path:
            args += ["-c", cfg.checkpoint_path]
        args += ["--device", cfg.device]
        args += ["--render" if cfg.with_rendering else "--no-render"]

    rendering_enabled = cfg.mode == "render" or (cfg.mode == "predict" and cfg.with_rendering)
    if rendering_enabled:
        args += ["--fps", str(cfg.fps)]

        args += ["--frames" if cfg.write_frames else "--no-frames"]
        if cfg.write_frames:
            args += [
                "--frame-ext",
                cfg.frame_ext,
                "--jpeg-quality",
                str(cfg.jpeg_quality),
            ]
        else:
            args += [
                "--video-ext",
                cfg.video_ext,
                "--macro-block-size",
                str(cfg.macro_block_size),
            ]
            if cfg.codec:
                args += ["--codec", cfg.codec]
            if cfg.bitrate:
                args += ["--bitrate", cfg.bitrate]

        args += [
            "--duration-scale",
            str(cfg.duration_scale),
            "--low-pass-filter-eps",
            str(cfg.low_pass_filter_eps),
            "--cuda-device",
            str(cfg.cuda_device),
            "--overwrite" if cfg.overwrite else "--no-overwrite",
            "--depth" if cfg.render_depth else "--no-depth",
        ]
        if cfg.trajectory_type:
            args += ["--trajectory-type", cfg.trajectory_type]
        if cfg.lookat_mode:
            args += ["--lookat-mode", cfg.lookat_mode]
        if cfg.max_disparity is not None:
            args += ["--max-disparity", str(cfg.max_disparity)]
        if cfg.max_zoom is not None:
            args += ["--max-zoom", str(cfg.max_zoom)]
        if cfg.distance_m is not None:
            args += ["--distance-m", str(cfg.distance_m)]
        if cfg.num_steps is not None:
            args += ["--num-steps", str(cfg.num_steps)]
        if cfg.num_repeats is not None:
            args += ["--num-repeats", str(cfg.num_repeats)]

        if cfg.output_prefix:
            args += ["--output-prefix", cfg.output_prefix]

        if cfg.trajectory_variants and cfg.trajectory_variants_count > 1:
            args += [
                "--trajectory-variants",
                "--trajectory-variants-count",
                str(cfg.trajectory_variants_count),
            ]
        else:
            args += ["--no-trajectory-variants"]

        if not cfg.progress:
            args += ["--no-progress"]

    if cfg.verbose:
        args += ["--verbose"]

    if cfg.extra_args.strip():
        args += shlex.split(cfg.extra_args)

    return args


def _get_cli_help_text(command: str) -> str:
    ctx = click.Context(main_cli)
    cmd = main_cli.get_command(ctx, command)
    if cmd is None:
        return "Command not found."
    return cmd.get_help(click.Context(cmd))


def main() -> None:
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("SHARP (GPU) Launcher")
    root.geometry("1000x720")

    output_q: queue.Queue[str] = queue.Queue()
    is_running = tk.BooleanVar(value=False)

    def parse_float(s: str, *, default: float) -> float:
        s = (s or "").strip()
        if not s:
            return default
        return float(s)

    def parse_int(s: str, *, default: int) -> int:
        s = (s or "").strip()
        if not s:
            return default
        return int(s)

    def parse_float_optional(s: str) -> float | None:
        s = (s or "").strip()
        return None if not s else float(s)

    def parse_int_optional(s: str) -> int | None:
        s = (s or "").strip()
        return None if not s else int(s)

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
        previz_btn.configure(state=("disabled" if running else "normal"))
        copy_btn.configure(state=("disabled" if running else "normal"))

    def browse_input_file() -> None:
        workflow = workflow_var.get()
        initial_dir = "Z:/Splats/images"
        initialdir_arg = {"initialdir": initial_dir} if Path(initial_dir).exists() else {}

        if workflow in {"Predict (PLY only)", "Predict + Render (Video)"}:
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

    def browse_input_folder() -> None:
        initial_dir = "Z:/Splats/images"
        initialdir_arg = {"initialdir": initial_dir} if Path(initial_dir).exists() else {}
        p = filedialog.askdirectory(title="Select an input folder", **initialdir_arg)
        if p:
            input_var.set(p)

    def browse_output_folder() -> None:
        default_out = output_var.get().strip() or "C:/Users/phill/Dropbox/Splats/outputs"
        initialdir_arg = {"initialdir": default_out} if Path(default_out).exists() else {}
        p = filedialog.askdirectory(title="Select an output folder", **initialdir_arg)
        if p:
            output_var.set(p)

    def open_output_folder() -> None:
        p = output_var.get().strip()
        if not p:
            return
        try:
            os.startfile(p)  # type: ignore[attr-defined]
        except Exception as exc:
            log_line(f"Could not open output folder: {exc}")

    def browse_checkpoint() -> None:
        p = filedialog.askopenfilename(
            title="Select a checkpoint (.pt) (optional)",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All", "*.*")],
        )
        if p:
            checkpoint_var.set(p)

    def show_help() -> None:
        mode = "render" if workflow_var.get() == "Render from PLY" else "predict"
        txt = _get_cli_help_text(mode)
        log_line("\n" + txt + "\n")

    def make_config(*, previz: bool) -> RunConfig | None:
        in_path = input_var.get().strip()
        out_path = output_var.get().strip()
        if not in_path or not out_path:
            log_line("Please select both input and output paths.")
            return None

        workflow = workflow_var.get()
        mode = "render" if workflow == "Render from PLY" else "predict"
        with_rendering = workflow == "Predict + Render (Video)"
        is_render_workflow = mode == "render" or with_rendering

        ckpt = checkpoint_var.get().strip() or None
        if use_default_checkpoint_var.get():
            ckpt = None

        variants_count = int(variants_count_var.get() or "1")
        variants_enabled = (
            is_render_workflow and bool(variants_enabled_var.get()) and variants_count > 1
        )

        if previz:
            if not is_render_workflow:
                log_line("Previz is only available for rendering workflows.")
                return None
            fps = 30.0
            num_steps = 30
            output_prefix = "previz_"
            render_depth = False
            variants_enabled = False
            variants_count = 1
        else:
            fps = parse_float(fps_var.get(), default=60.0)
            num_steps = parse_int_optional(num_steps_var.get())
            output_prefix = output_prefix_var.get().strip()
            render_depth = bool(render_depth_var.get())

        cfg = RunConfig(
            mode=mode,
            input_path=in_path,
            output_path=out_path,
            checkpoint_path=ckpt,
            device=device_var.get().strip() or "default",
            with_rendering=(with_rendering or previz) if mode == "predict" else False,
            fps=fps,
            write_frames=bool(write_frames_var.get()),
            frame_ext=(frame_ext_var.get().strip() or "png"),
            jpeg_quality=parse_int(jpeg_quality_var.get(), default=92),
            video_ext=video_ext_var.get().strip() or "mp4",
            codec=(codec_var.get().strip() or None),
            bitrate=(bitrate_var.get().strip() or None),
            macro_block_size=parse_int(macro_block_size_var.get(), default=16),
            render_depth=render_depth,
            duration_scale=parse_float(duration_scale_var.get(), default=8.0),
            trajectory_type=(trajectory_type_var.get().strip() or None),
            lookat_mode=(lookat_mode_var.get().strip() or None),
            max_disparity=parse_float_optional(max_disparity_var.get()),
            max_zoom=parse_float_optional(max_zoom_var.get()),
            distance_m=parse_float_optional(distance_m_var.get()),
            num_steps=num_steps,
            num_repeats=parse_int_optional(num_repeats_var.get()),
            trajectory_variants=variants_enabled,
            trajectory_variants_count=variants_count,
            output_prefix=output_prefix,
            overwrite=bool(overwrite_var.get()),
            low_pass_filter_eps=parse_float(low_pass_filter_eps_var.get(), default=0.0),
            cuda_device=parse_int(cuda_device_var.get(), default=0),
            progress=bool(progress_var.get()),
            progress_style=progress_style_var.get(),
            verbose=bool(verbose_var.get()),
            disable_truststore=bool(disable_truststore_var.get()),
            extra_args=extra_var.get(),
        )
        return cfg

    def copy_command() -> None:
        cfg = make_config(previz=False)
        if cfg is None:
            return
        cmd = f"sharp {' '.join(_build_args(cfg))}"
        root.clipboard_clear()
        root.clipboard_append(cmd)
        log_line("Copied command to clipboard.")

    def run_in_thread(cfg: RunConfig) -> None:
        set_running(True)
        started = time.time()
        try:
            args = _build_args(cfg)
            log_line(f"Running: sharp {' '.join(args)}")

            old_progress_style = os.environ.get("SHARP_PROGRESS_STYLE")
            old_disable_truststore = os.environ.get("SHARP_DISABLE_TRUSTSTORE")
            try:
                if cfg.progress_style == "text":
                    os.environ["SHARP_PROGRESS_STYLE"] = "text"
                else:
                    os.environ.pop("SHARP_PROGRESS_STYLE", None)

                if cfg.disable_truststore:
                    os.environ["SHARP_DISABLE_TRUSTSTORE"] = "1"
                else:
                    os.environ.pop("SHARP_DISABLE_TRUSTSTORE", None)

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
                        "the checkpoint and select it via 'Checkpoint (.pt)'."
                    )
                raise
            finally:
                if old_progress_style is None:
                    os.environ.pop("SHARP_PROGRESS_STYLE", None)
                else:
                    os.environ["SHARP_PROGRESS_STYLE"] = old_progress_style
                if old_disable_truststore is None:
                    os.environ.pop("SHARP_DISABLE_TRUSTSTORE", None)
                else:
                    os.environ["SHARP_DISABLE_TRUSTSTORE"] = old_disable_truststore

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
        cfg = make_config(previz=False)
        if cfg is None:
            return
        threading.Thread(target=run_in_thread, args=(cfg,), daemon=True).start()

    def on_previz() -> None:
        if is_running.get():
            return
        cfg = make_config(previz=True)
        if cfg is None:
            return
        log_line("Previz: 30 fps, 30 frames (~1s), single trajectory, no depth.")
        threading.Thread(target=run_in_thread, args=(cfg,), daemon=True).start()

    def on_show_flags() -> None:
        show_help()

    def on_clear_log() -> None:
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")

    def update_mode_ui(*_args: object) -> None:
        workflow = workflow_var.get()
        is_predict = workflow in {"Predict (PLY only)", "Predict + Render (Video)"}
        is_render = workflow in {"Predict + Render (Video)", "Render from PLY"}

        device_combo.configure(state=("readonly" if is_predict else "disabled"))
        use_default_checkpoint_chk.configure(state=("normal" if is_predict else "disabled"))
        disable_truststore_chk.configure(state=("normal" if is_predict else "disabled"))

        checkpoint_state = "normal" if is_predict and not use_default_checkpoint_var.get() else "disabled"
        checkpoint_entry.configure(state=checkpoint_state)
        checkpoint_browse_btn.configure(state=checkpoint_state)

        render_state = "normal" if is_render else "disabled"
        common_render_widgets = (
            fps_entry,
            write_frames_chk,
            frame_ext_combo,
            jpeg_quality_entry,
            render_depth_chk,
            low_pass_entry,
            cuda_device_entry,
            duration_scale_entry,
            trajectory_type_combo,
            lookat_mode_combo,
            max_disparity_entry,
            max_zoom_entry,
            distance_m_entry,
            num_steps_entry,
            num_repeats_entry,
            variants_enabled_chk,
            variants_count_combo,
        )
        video_only_widgets = (
            video_ext_combo,
            codec_entry,
            bitrate_entry,
            macro_block_entry,
        )

        for w in (*common_render_widgets, *video_only_widgets):
            w.configure(state=render_state)

        if render_state == "normal":
            frames_enabled = bool(write_frames_var.get())

            for w in video_only_widgets:
                w.configure(state=("disabled" if frames_enabled else "normal"))

            frame_state = "normal" if frames_enabled else "disabled"
            frame_ext_combo.configure(state=frame_state)
            if frames_enabled and frame_ext_var.get().strip().lower() == "jpg":
                jpeg_quality_entry.configure(state="normal")
            else:
                jpeg_quality_entry.configure(state="disabled")

        previz_btn.configure(state=("normal" if is_render and not is_running.get() else "disabled"))

    def update_checkpoint_ui(*_args: object) -> None:
        update_mode_ui()

    # Layout
    outer = ttk.Frame(root, padding=10)
    outer.pack(fill="both", expand=True)

    workflow_var = tk.StringVar(value="Predict + Render (Video)")
    input_var = tk.StringVar()
    output_var = tk.StringVar(value="C:/Users/phill/Dropbox/Splats/outputs")
    output_prefix_var = tk.StringVar(value="")
    overwrite_var = tk.IntVar(value=1)

    device_var = tk.StringVar(value="cuda")
    checkpoint_var = tk.StringVar(value="")
    use_default_checkpoint_var = tk.IntVar(value=1)
    disable_truststore_var = tk.IntVar(value=0)

    fps_var = tk.StringVar(value="60")
    write_frames_var = tk.IntVar(value=0)
    frame_ext_var = tk.StringVar(value="png")
    jpeg_quality_var = tk.StringVar(value="92")
    video_ext_var = tk.StringVar(value="mp4")
    codec_var = tk.StringVar(value="")
    bitrate_var = tk.StringVar(value="")
    macro_block_size_var = tk.StringVar(value="16")
    render_depth_var = tk.IntVar(value=1)
    low_pass_filter_eps_var = tk.StringVar(value="0.0")
    cuda_device_var = tk.StringVar(value="0")

    duration_scale_var = tk.StringVar(value="8")
    trajectory_type_var = tk.StringVar(value="")
    lookat_mode_var = tk.StringVar(value="")
    max_disparity_var = tk.StringVar(value="")
    max_zoom_var = tk.StringVar(value="")
    distance_m_var = tk.StringVar(value="")
    num_steps_var = tk.StringVar(value="")
    num_repeats_var = tk.StringVar(value="")
    variants_enabled_var = tk.IntVar(value=1)
    variants_count_var = tk.StringVar(value="5")

    progress_var = tk.IntVar(value=1)
    progress_style_var = tk.StringVar(value="text")
    verbose_var = tk.IntVar(value=0)
    extra_var = tk.StringVar(value="")

    notebook = ttk.Notebook(outer)
    notebook.grid(row=0, column=0, sticky="nsew")
    outer.rowconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)

    tab_io = ttk.Frame(notebook, padding=10)
    tab_model = ttk.Frame(notebook, padding=10)
    tab_traj = ttk.Frame(notebook, padding=10)
    tab_render = ttk.Frame(notebook, padding=10)
    tab_adv = ttk.Frame(notebook, padding=10)

    notebook.add(tab_io, text="Input/Output")
    notebook.add(tab_model, text="Model")
    notebook.add(tab_traj, text="Trajectory")
    notebook.add(tab_render, text="Render/Video")
    notebook.add(tab_adv, text="Advanced")

    # Input/Output tab
    r = 0
    ttk.Label(tab_io, text="Workflow").grid(row=r, column=0, sticky="w")
    ttk.Combobox(
        tab_io,
        textvariable=workflow_var,
        values=("Predict + Render (Video)", "Predict (PLY only)", "Render from PLY"),
        state="readonly",
        width=28,
    ).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(tab_io, text="Input").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_io, textvariable=input_var, width=74).grid(
        row=r, column=1, columnspan=3, sticky="we", pady=(10, 0)
    )
    ttk.Button(tab_io, text="Browse File…", command=browse_input_file).grid(
        row=r, column=4, padx=(8, 0), pady=(10, 0)
    )
    ttk.Button(tab_io, text="Browse Folder…", command=browse_input_folder).grid(
        row=r, column=5, padx=(8, 0), pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_io, text="Output folder").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_io, textvariable=output_var, width=74).grid(
        row=r, column=1, columnspan=3, sticky="we", pady=(10, 0)
    )
    ttk.Button(tab_io, text="Browse…", command=browse_output_folder).grid(
        row=r, column=4, padx=(8, 0), pady=(10, 0)
    )
    ttk.Button(tab_io, text="Open", command=open_output_folder).grid(
        row=r, column=5, padx=(8, 0), pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_io, text="Output prefix").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_io, textvariable=output_prefix_var, width=20).grid(
        row=r, column=1, sticky="w", pady=(10, 0)
    )
    ttk.Checkbutton(tab_io, text="Overwrite outputs", variable=overwrite_var).grid(
        row=r, column=2, columnspan=2, sticky="w", pady=(10, 0)
    )
    tab_io.columnconfigure(1, weight=1)

    # Model tab
    r = 0
    ttk.Label(tab_model, text="Predict device").grid(row=r, column=0, sticky="w")
    device_combo = ttk.Combobox(
        tab_model,
        textvariable=device_var,
        values=("default", "cuda", "cpu", "mps"),
        state="readonly",
        width=12,
    )
    device_combo.grid(row=r, column=1, sticky="w")

    r += 1
    use_default_checkpoint_chk = ttk.Checkbutton(
        tab_model,
        text="Use default checkpoint (auto download)",
        variable=use_default_checkpoint_var,
    )
    use_default_checkpoint_chk.grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 0))

    r += 1
    ttk.Label(tab_model, text="Checkpoint (.pt)").grid(row=r, column=0, sticky="w", pady=(10, 0))
    checkpoint_entry = ttk.Entry(tab_model, textvariable=checkpoint_var, width=60)
    checkpoint_entry.grid(row=r, column=1, sticky="we", pady=(10, 0))
    checkpoint_browse_btn = ttk.Button(tab_model, text="Browse…", command=browse_checkpoint)
    checkpoint_browse_btn.grid(row=r, column=2, padx=(8, 0), pady=(10, 0))

    r += 1
    disable_truststore_chk = ttk.Checkbutton(
        tab_model,
        text="Disable Windows trust store (SSL)",
        variable=disable_truststore_var,
    )
    disable_truststore_chk.grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 0))
    tab_model.columnconfigure(1, weight=1)

    # Trajectory tab
    r = 0
    ttk.Label(tab_traj, text="Duration scale").grid(row=r, column=0, sticky="w")
    duration_scale_entry = ttk.Entry(tab_traj, textvariable=duration_scale_var, width=10)
    duration_scale_entry.grid(row=r, column=1, sticky="w")
    ttk.Label(tab_traj, text="Num steps (optional)").grid(row=r, column=2, sticky="w")
    num_steps_entry = ttk.Entry(tab_traj, textvariable=num_steps_var, width=10)
    num_steps_entry.grid(row=r, column=3, sticky="w")

    r += 1
    ttk.Label(tab_traj, text="Trajectory type").grid(row=r, column=0, sticky="w", pady=(10, 0))
    trajectory_type_combo = ttk.Combobox(
        tab_traj,
        textvariable=trajectory_type_var,
        values=("", "swipe", "shake", "rotate", "rotate_forward"),
        state="readonly",
        width=18,
    )
    trajectory_type_combo.grid(row=r, column=1, sticky="w", pady=(10, 0))
    ttk.Label(tab_traj, text="Look-at").grid(row=r, column=2, sticky="w", pady=(10, 0))
    lookat_mode_combo = ttk.Combobox(
        tab_traj,
        textvariable=lookat_mode_var,
        values=("", "point", "ahead"),
        state="readonly",
        width=12,
    )
    lookat_mode_combo.grid(row=r, column=3, sticky="w", pady=(10, 0))

    r += 1
    ttk.Label(tab_traj, text="Max disparity").grid(row=r, column=0, sticky="w", pady=(10, 0))
    max_disparity_entry = ttk.Entry(tab_traj, textvariable=max_disparity_var, width=10)
    max_disparity_entry.grid(row=r, column=1, sticky="w", pady=(10, 0))
    ttk.Label(tab_traj, text="Max zoom").grid(row=r, column=2, sticky="w", pady=(10, 0))
    max_zoom_entry = ttk.Entry(tab_traj, textvariable=max_zoom_var, width=10)
    max_zoom_entry.grid(row=r, column=3, sticky="w", pady=(10, 0))

    r += 1
    ttk.Label(tab_traj, text="Distance (m)").grid(row=r, column=0, sticky="w", pady=(10, 0))
    distance_m_entry = ttk.Entry(tab_traj, textvariable=distance_m_var, width=10)
    distance_m_entry.grid(row=r, column=1, sticky="w", pady=(10, 0))
    ttk.Label(tab_traj, text="Num repeats").grid(row=r, column=2, sticky="w", pady=(10, 0))
    num_repeats_entry = ttk.Entry(tab_traj, textvariable=num_repeats_var, width=10)
    num_repeats_entry.grid(row=r, column=3, sticky="w", pady=(10, 0))

    r += 1
    variants_enabled_chk = ttk.Checkbutton(
        tab_traj, text="Trajectory variants", variable=variants_enabled_var
    )
    variants_enabled_chk.grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Label(tab_traj, text="Count").grid(row=r, column=1, sticky="e", pady=(10, 0))
    variants_count_combo = ttk.Combobox(
        tab_traj,
        textvariable=variants_count_var,
        values=("1", "2", "3", "4", "5"),
        state="readonly",
        width=6,
    )
    variants_count_combo.grid(row=r, column=2, sticky="w", pady=(10, 0))

    # Render tab
    r = 0
    ttk.Label(tab_render, text="FPS").grid(row=r, column=0, sticky="w")
    fps_entry = ttk.Entry(tab_render, textvariable=fps_var, width=10)
    fps_entry.grid(row=r, column=1, sticky="w")
    ttk.Label(tab_render, text="Video ext").grid(row=r, column=2, sticky="w")
    video_ext_combo = ttk.Combobox(
        tab_render, textvariable=video_ext_var, values=("mp4", "mov"), state="readonly", width=8
    )
    video_ext_combo.grid(row=r, column=3, sticky="w")

    r += 1
    write_frames_chk = ttk.Checkbutton(
        tab_render, text="Output frames (image sequence)", variable=write_frames_var
    )
    write_frames_chk.grid(row=r, column=0, columnspan=2, sticky="w", pady=(10, 0))
    ttk.Label(tab_render, text="Frame ext").grid(row=r, column=2, sticky="w", pady=(10, 0))
    frame_ext_combo = ttk.Combobox(
        tab_render, textvariable=frame_ext_var, values=("png", "jpg"), state="readonly", width=8
    )
    frame_ext_combo.grid(row=r, column=3, sticky="w", pady=(10, 0))

    r += 1
    ttk.Label(tab_render, text="JPEG quality").grid(row=r, column=2, sticky="w", pady=(10, 0))
    jpeg_quality_entry = ttk.Entry(tab_render, textvariable=jpeg_quality_var, width=10)
    jpeg_quality_entry.grid(row=r, column=3, sticky="w", pady=(10, 0))

    r += 1
    ttk.Label(tab_render, text="Codec (optional)").grid(row=r, column=0, sticky="w", pady=(10, 0))
    codec_entry = ttk.Entry(tab_render, textvariable=codec_var, width=16)
    codec_entry.grid(row=r, column=1, sticky="w", pady=(10, 0))
    ttk.Label(tab_render, text="Bitrate (optional)").grid(row=r, column=2, sticky="w", pady=(10, 0))
    bitrate_entry = ttk.Entry(tab_render, textvariable=bitrate_var, width=16)
    bitrate_entry.grid(row=r, column=3, sticky="w", pady=(10, 0))

    r += 1
    ttk.Label(tab_render, text="Macro block size").grid(row=r, column=0, sticky="w", pady=(10, 0))
    macro_block_entry = ttk.Entry(tab_render, textvariable=macro_block_size_var, width=10)
    macro_block_entry.grid(row=r, column=1, sticky="w", pady=(10, 0))
    render_depth_chk = ttk.Checkbutton(
        tab_render, text="Render depth output", variable=render_depth_var
    )
    render_depth_chk.grid(row=r, column=2, columnspan=2, sticky="w", pady=(10, 0))

    r += 1
    ttk.Label(tab_render, text="Low-pass eps").grid(row=r, column=0, sticky="w", pady=(10, 0))
    low_pass_entry = ttk.Entry(tab_render, textvariable=low_pass_filter_eps_var, width=10)
    low_pass_entry.grid(row=r, column=1, sticky="w", pady=(10, 0))
    ttk.Label(tab_render, text="CUDA device").grid(row=r, column=2, sticky="w", pady=(10, 0))
    cuda_device_entry = ttk.Entry(tab_render, textvariable=cuda_device_var, width=10)
    cuda_device_entry.grid(row=r, column=3, sticky="w", pady=(10, 0))

    # Advanced tab
    r = 0
    ttk.Checkbutton(tab_adv, text="Progress", variable=progress_var).grid(row=r, column=0, sticky="w")
    ttk.Label(tab_adv, text="Progress style").grid(row=r, column=1, sticky="w")
    ttk.Combobox(
        tab_adv,
        textvariable=progress_style_var,
        values=("text", "auto"),
        state="readonly",
        width=10,
    ).grid(row=r, column=2, sticky="w")
    ttk.Checkbutton(tab_adv, text="Verbose", variable=verbose_var).grid(row=r, column=3, sticky="w")

    r += 1
    ttk.Label(tab_adv, text="Extra args").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_adv, textvariable=extra_var, width=82).grid(
        row=r, column=1, columnspan=3, sticky="we", pady=(10, 0)
    )
    tab_adv.columnconfigure(1, weight=1)

    # Log (always visible under tabs)
    log_frame = ttk.LabelFrame(outer, text="Log")
    log_frame.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
    log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
    log_text.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
    scrollbar = ttk.Scrollbar(log_frame, command=log_text.yview)
    scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
    log_text.configure(yscrollcommand=scrollbar.set)
    log_frame.rowconfigure(0, weight=1)
    log_frame.columnconfigure(0, weight=1)

    # Bottom buttons
    buttons = ttk.Frame(outer, padding=(0, 10, 0, 0))
    buttons.grid(row=2, column=0, sticky="we")
    run_btn = ttk.Button(buttons, text="Run", command=on_run)
    run_btn.pack(side="left")
    previz_btn = ttk.Button(buttons, text="Previz (fast)", command=on_previz)
    previz_btn.pack(side="left", padx=(8, 0))
    copy_btn = ttk.Button(buttons, text="Copy Command", command=copy_command)
    copy_btn.pack(side="left", padx=(8, 0))
    ttk.Button(buttons, text="Show Flags/Help", command=on_show_flags).pack(
        side="left", padx=(8, 0)
    )
    ttk.Button(buttons, text="Clear Log", command=on_clear_log).pack(side="left", padx=(8, 0))

    use_default_checkpoint_var.trace_add("write", update_checkpoint_ui)
    workflow_var.trace_add("write", update_mode_ui)
    write_frames_var.trace_add("write", update_mode_ui)
    frame_ext_var.trace_add("write", update_mode_ui)
    is_running.trace_add("write", update_mode_ui)
    update_mode_ui()

    drain_output()
    log_line("Tip: Use 'Copy Command' to run the same settings in a terminal.")
    root.mainloop()
