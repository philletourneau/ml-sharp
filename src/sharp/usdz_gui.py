"""Windows GUI launcher for SHARP mesh/USDZ workflows."""

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
    command: str  # "mesh" or "usdz"
    args: list[str]
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
    root.title("SHARP Mesh / USDZ Launcher")
    root.geometry("1020x740")

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

    def browse_input() -> None:
        initial_dir = output_folder_var.get().strip() or "Z:/Splats/output"
        initialdir_arg = {"initialdir": initial_dir} if Path(initial_dir).exists() else {}
        p = filedialog.askopenfilename(
            title="Select a .ply file",
            filetypes=[("PLY", "*.ply"), ("All", "*.*")],
            **initialdir_arg,
        )
        if p:
            input_path_var.set(p)
            if not base_name_var.get().strip():
                base_name_var.set(Path(p).stem)

    def browse_output_folder() -> None:
        default_out = output_folder_var.get().strip() or "Z:/Splats/output"
        initialdir_arg = {"initialdir": default_out} if Path(default_out).exists() else {}
        p = filedialog.askdirectory(title="Select an output folder", **initialdir_arg)
        if p:
            output_folder_var.set(p)

    def open_output_folder() -> None:
        p = output_folder_var.get().strip()
        if not p:
            return
        try:
            os.startfile(p)  # type: ignore[attr-defined]
        except Exception as exc:
            log_line(f"Could not open output folder: {exc}")

    def computed_paths() -> tuple[Path, Path]:
        out_dir = Path(output_folder_var.get().strip() or "Z:/Splats/output")
        base = base_name_var.get().strip() or "scene"
        mesh_path = out_dir / f"{base}_mesh.ply"
        usdz_path = out_dir / f"{base}.usdz"
        return mesh_path, usdz_path

    def update_path_preview(*_args: object) -> None:
        mesh_path, usdz_path = computed_paths()
        mesh_preview_var.set(str(mesh_path))
        usdz_preview_var.set(str(usdz_path))

    def build_run_config() -> RunConfig | None:
        in_path = input_path_var.get().strip()
        out_folder = output_folder_var.get().strip()
        if not in_path or not out_folder:
            log_line("Please select both input and output paths.")
            return None

        mesh_path, usdz_path = computed_paths()
        verbose = bool(verbose_var.get())

        workflow = workflow_var.get()
        if workflow == "Gaussians (.ply) -> Mesh (.ply)":
            args = [
                "mesh",
                "-i",
                in_path,
                "-o",
                str(mesh_path),
                "--max-render-dim",
                str(parse_int(max_render_dim_var.get(), default=640)),
                "--depth-trunc",
                str(parse_float(depth_trunc_var.get(), default=10.0)),
                "--voxel-length",
                str(parse_float(voxel_length_var.get(), default=0.01)),
                "--sdf-trunc",
                str(parse_float(sdf_trunc_var.get(), default=0.04)),
                "--trajectory-type",
                (trajectory_type_var.get().strip() or "rotate"),
                "--lookat-mode",
                (lookat_mode_var.get().strip() or "point"),
                "--num-steps",
                str(parse_int(num_steps_var.get(), default=240)),
                "--num-repeats",
                str(parse_int(num_repeats_var.get(), default=1)),
                "--low-pass-filter-eps",
                str(parse_float(low_pass_eps_var.get(), default=0.0)),
                "--cuda-device",
                str(parse_int(cuda_device_var.get(), default=0)),
            ]
            if parse_float_optional(max_disparity_var.get()) is not None:
                args += ["--max-disparity", str(parse_float_optional(max_disparity_var.get()))]
            if parse_float_optional(max_zoom_var.get()) is not None:
                args += ["--max-zoom", str(parse_float_optional(max_zoom_var.get()))]
            if parse_float_optional(distance_m_var.get()) is not None:
                args += ["--distance-m", str(parse_float_optional(distance_m_var.get()))]
            if target_triangles_var.get().strip():
                args += ["--target-triangles", str(parse_int(target_triangles_var.get(), default=0))]
            if not bool(progress_var.get()):
                args += ["--no-progress"]
            if verbose:
                args += ["--verbose"]
            return RunConfig(command="mesh", args=args, verbose=verbose)

        args = [
            "usdz",
            "-i",
            in_path,
            "-o",
            str(usdz_path),
            "--root-name",
            (root_name_var.get().strip() or "Root"),
            "--mesh-name",
            (mesh_name_var.get().strip() or "Mesh"),
            "--coordinate-system",
            (coord_var.get().strip() or "usd"),
        ]

        if workflow == "Gaussians (.ply) -> USDZ":
            if bool(write_mesh_var.get()):
                args += ["--mesh-output", str(mesh_path)]
            args += [
                "--max-render-dim",
                str(parse_int(max_render_dim_var.get(), default=640)),
                "--depth-trunc",
                str(parse_float(depth_trunc_var.get(), default=10.0)),
                "--voxel-length",
                str(parse_float(voxel_length_var.get(), default=0.01)),
                "--sdf-trunc",
                str(parse_float(sdf_trunc_var.get(), default=0.04)),
                "--trajectory-type",
                (trajectory_type_var.get().strip() or "rotate"),
                "--lookat-mode",
                (lookat_mode_var.get().strip() or "point"),
                "--num-steps",
                str(parse_int(num_steps_var.get(), default=240)),
                "--num-repeats",
                str(parse_int(num_repeats_var.get(), default=1)),
                "--low-pass-filter-eps",
                str(parse_float(low_pass_eps_var.get(), default=0.0)),
                "--cuda-device",
                str(parse_int(cuda_device_var.get(), default=0)),
            ]
            if parse_float_optional(max_disparity_var.get()) is not None:
                args += ["--max-disparity", str(parse_float_optional(max_disparity_var.get()))]
            if parse_float_optional(max_zoom_var.get()) is not None:
                args += ["--max-zoom", str(parse_float_optional(max_zoom_var.get()))]
            if parse_float_optional(distance_m_var.get()) is not None:
                args += ["--distance-m", str(parse_float_optional(distance_m_var.get()))]
            if not bool(progress_var.get()):
                args += ["--no-progress"]

        if verbose:
            args += ["--verbose"]

        if extra_args_var.get().strip():
            args += shlex.split(extra_args_var.get())

        return RunConfig(command="usdz", args=args, verbose=verbose)

    def copy_command() -> None:
        cfg = build_run_config()
        if cfg is None:
            return
        cmd = "sharp " + " ".join(cfg.args)
        root.clipboard_clear()
        root.clipboard_append(cmd)
        log_line("Copied command to clipboard.")

    def show_help() -> None:
        workflow = workflow_var.get()
        cmd = "mesh" if workflow == "Gaussians (.ply) -> Mesh (.ply)" else "usdz"
        log_line("\n" + _get_cli_help_text(cmd) + "\n")

    def run_in_thread(cfg: RunConfig) -> None:
        set_running(True)
        started = time.time()
        try:
            log_line("Running: sharp " + " ".join(cfg.args))
            with contextlib.redirect_stdout(_QueueWriter(output_q)), contextlib.redirect_stderr(
                _QueueWriter(output_q)
            ):
                main_cli.main(args=cfg.args, standalone_mode=False)
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
        cfg = build_run_config()
        if cfg is None:
            return
        threading.Thread(target=run_in_thread, args=(cfg,), daemon=True).start()

    def on_clear_log() -> None:
        log_text.configure(state="normal")
        log_text.delete("1.0", "end")
        log_text.configure(state="disabled")

    # UI state vars
    workflow_var = tk.StringVar(value="Gaussians (.ply) -> USDZ")
    input_path_var = tk.StringVar()
    output_folder_var = tk.StringVar(value="Z:/Splats/output")
    base_name_var = tk.StringVar(value="")
    write_mesh_var = tk.IntVar(value=1)

    mesh_preview_var = tk.StringVar(value="")
    usdz_preview_var = tk.StringVar(value="")

    # Meshing options
    max_render_dim_var = tk.StringVar(value="640")
    depth_trunc_var = tk.StringVar(value="10.0")
    voxel_length_var = tk.StringVar(value="0.01")
    sdf_trunc_var = tk.StringVar(value="0.04")
    trajectory_type_var = tk.StringVar(value="rotate")
    lookat_mode_var = tk.StringVar(value="point")
    max_disparity_var = tk.StringVar(value="")
    max_zoom_var = tk.StringVar(value="")
    distance_m_var = tk.StringVar(value="")
    num_steps_var = tk.StringVar(value="240")
    num_repeats_var = tk.StringVar(value="1")
    low_pass_eps_var = tk.StringVar(value="0.0")
    cuda_device_var = tk.StringVar(value="0")
    target_triangles_var = tk.StringVar(value="")

    # USD options
    root_name_var = tk.StringVar(value="Root")
    mesh_name_var = tk.StringVar(value="Mesh")
    coord_var = tk.StringVar(value="usd")

    # Advanced
    progress_var = tk.IntVar(value=1)
    verbose_var = tk.IntVar(value=0)
    extra_args_var = tk.StringVar(value="")

    # Layout
    outer = ttk.Frame(root, padding=10)
    outer.pack(fill="both", expand=True)

    notebook = ttk.Notebook(outer)
    notebook.grid(row=0, column=0, sticky="nsew")
    outer.rowconfigure(0, weight=1)
    outer.columnconfigure(0, weight=1)

    tab_io = ttk.Frame(notebook, padding=10)
    tab_mesh = ttk.Frame(notebook, padding=10)
    tab_usd = ttk.Frame(notebook, padding=10)
    tab_adv = ttk.Frame(notebook, padding=10)
    tab_log = ttk.Frame(notebook, padding=10)

    notebook.add(tab_io, text="Input/Output")
    notebook.add(tab_mesh, text="Meshing")
    notebook.add(tab_usd, text="USDZ")
    notebook.add(tab_adv, text="Advanced")
    notebook.add(tab_log, text="Log")

    # Input/Output
    r = 0
    ttk.Label(tab_io, text="Workflow").grid(row=r, column=0, sticky="w")
    ttk.Combobox(
        tab_io,
        textvariable=workflow_var,
        values=("Gaussians (.ply) -> USDZ", "Mesh (.ply) -> USDZ", "Gaussians (.ply) -> Mesh (.ply)"),
        state="readonly",
        width=26,
    ).grid(row=r, column=1, sticky="w")

    r += 1
    ttk.Label(tab_io, text="Input .ply").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_io, textvariable=input_path_var, width=78).grid(
        row=r, column=1, columnspan=2, sticky="we", pady=(10, 0)
    )
    ttk.Button(tab_io, text="Browse…", command=browse_input).grid(
        row=r, column=3, padx=(8, 0), pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_io, text="Output folder").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_io, textvariable=output_folder_var, width=78).grid(
        row=r, column=1, columnspan=2, sticky="we", pady=(10, 0)
    )
    ttk.Button(tab_io, text="Browse…", command=browse_output_folder).grid(
        row=r, column=3, padx=(8, 0), pady=(10, 0)
    )
    ttk.Button(tab_io, text="Open", command=open_output_folder).grid(
        row=r, column=4, padx=(8, 0), pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_io, text="Base name").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_io, textvariable=base_name_var, width=24).grid(
        row=r, column=1, sticky="w", pady=(10, 0)
    )
    ttk.Checkbutton(tab_io, text="Write mesh PLY", variable=write_mesh_var).grid(
        row=r, column=2, sticky="w", pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_io, text="Mesh output").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_io, textvariable=mesh_preview_var, state="readonly", width=78).grid(
        row=r, column=1, columnspan=3, sticky="we", pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_io, text="USDZ output").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_io, textvariable=usdz_preview_var, state="readonly", width=78).grid(
        row=r, column=1, columnspan=3, sticky="we", pady=(10, 0)
    )

    tab_io.columnconfigure(1, weight=1)
    output_folder_var.trace_add("write", update_path_preview)
    base_name_var.trace_add("write", update_path_preview)
    update_path_preview()

    # Meshing tab
    r = 0
    ttk.Label(tab_mesh, text="Max render dim").grid(row=r, column=0, sticky="w")
    ttk.Entry(tab_mesh, textvariable=max_render_dim_var, width=10).grid(row=r, column=1, sticky="w")
    ttk.Label(tab_mesh, text="Depth trunc (m)").grid(row=r, column=2, sticky="w")
    ttk.Entry(tab_mesh, textvariable=depth_trunc_var, width=10).grid(row=r, column=3, sticky="w")

    r += 1
    ttk.Label(tab_mesh, text="Voxel length (m)").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=voxel_length_var, width=10).grid(
        row=r, column=1, sticky="w", pady=(10, 0)
    )
    ttk.Label(tab_mesh, text="SDF trunc (m)").grid(row=r, column=2, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=sdf_trunc_var, width=10).grid(
        row=r, column=3, sticky="w", pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_mesh, text="Trajectory").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Combobox(
        tab_mesh,
        textvariable=trajectory_type_var,
        values=("swipe", "shake", "rotate", "rotate_forward"),
        state="readonly",
        width=16,
    ).grid(row=r, column=1, sticky="w", pady=(10, 0))
    ttk.Label(tab_mesh, text="Look-at").grid(row=r, column=2, sticky="w", pady=(10, 0))
    ttk.Combobox(
        tab_mesh, textvariable=lookat_mode_var, values=("point", "ahead"), state="readonly", width=10
    ).grid(row=r, column=3, sticky="w", pady=(10, 0))

    r += 1
    ttk.Label(tab_mesh, text="Max disparity").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=max_disparity_var, width=10).grid(
        row=r, column=1, sticky="w", pady=(10, 0)
    )
    ttk.Label(tab_mesh, text="Max zoom").grid(row=r, column=2, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=max_zoom_var, width=10).grid(
        row=r, column=3, sticky="w", pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_mesh, text="Distance (m)").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=distance_m_var, width=10).grid(
        row=r, column=1, sticky="w", pady=(10, 0)
    )
    ttk.Label(tab_mesh, text="Views (steps)").grid(row=r, column=2, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=num_steps_var, width=10).grid(
        row=r, column=3, sticky="w", pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_mesh, text="Repeats").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=num_repeats_var, width=10).grid(
        row=r, column=1, sticky="w", pady=(10, 0)
    )
    ttk.Label(tab_mesh, text="Low-pass eps").grid(row=r, column=2, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=low_pass_eps_var, width=10).grid(
        row=r, column=3, sticky="w", pady=(10, 0)
    )

    r += 1
    ttk.Label(tab_mesh, text="CUDA device").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=cuda_device_var, width=10).grid(
        row=r, column=1, sticky="w", pady=(10, 0)
    )
    ttk.Label(tab_mesh, text="Target triangles").grid(row=r, column=2, sticky="w", pady=(10, 0))
    ttk.Entry(tab_mesh, textvariable=target_triangles_var, width=10).grid(
        row=r, column=3, sticky="w", pady=(10, 0)
    )

    # USDZ tab
    r = 0
    ttk.Label(tab_usd, text="USD root name").grid(row=r, column=0, sticky="w")
    ttk.Entry(tab_usd, textvariable=root_name_var, width=18).grid(row=r, column=1, sticky="w")
    ttk.Label(tab_usd, text="Mesh name").grid(row=r, column=2, sticky="w")
    ttk.Entry(tab_usd, textvariable=mesh_name_var, width=18).grid(row=r, column=3, sticky="w")

    r += 1
    ttk.Label(tab_usd, text="Coordinate system").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Combobox(
        tab_usd, textvariable=coord_var, values=("usd", "sharp"), state="readonly", width=10
    ).grid(row=r, column=1, sticky="w", pady=(10, 0))
    ttk.Label(tab_usd, text="(usd flips Y/Z for RealityKit)").grid(row=r, column=2, columnspan=2, sticky="w", pady=(10, 0))

    # Advanced tab
    r = 0
    ttk.Checkbutton(tab_adv, text="Progress", variable=progress_var).grid(row=r, column=0, sticky="w")
    ttk.Checkbutton(tab_adv, text="Verbose", variable=verbose_var).grid(row=r, column=1, sticky="w")
    r += 1
    ttk.Label(tab_adv, text="Extra args").grid(row=r, column=0, sticky="w", pady=(10, 0))
    ttk.Entry(tab_adv, textvariable=extra_args_var, width=86).grid(
        row=r, column=1, columnspan=3, sticky="we", pady=(10, 0)
    )
    tab_adv.columnconfigure(1, weight=1)

    # Log tab
    log_text = tk.Text(tab_log, height=26, wrap="word", state="disabled")
    log_text.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(tab_log, command=log_text.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    log_text.configure(yscrollcommand=scrollbar.set)
    tab_log.rowconfigure(0, weight=1)
    tab_log.columnconfigure(0, weight=1)

    # Bottom buttons
    buttons = ttk.Frame(outer, padding=(0, 10, 0, 0))
    buttons.grid(row=1, column=0, sticky="we")
    run_btn = ttk.Button(buttons, text="Run", command=on_run)
    run_btn.pack(side="left")
    copy_btn = ttk.Button(buttons, text="Copy Command", command=copy_command)
    copy_btn.pack(side="left", padx=(8, 0))
    ttk.Button(buttons, text="Show Flags/Help", command=show_help).pack(side="left", padx=(8, 0))
    ttk.Button(buttons, text="Clear Log", command=on_clear_log).pack(side="left", padx=(8, 0))

    drain_output()
    log_line("Tip: 'Mesh (.ply) -> USDZ' does not require CUDA.")
    root.mainloop()

