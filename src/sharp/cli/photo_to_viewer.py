"""CLI pipeline: photo -> SHARP Gaussians PLY -> self-contained splat viewer HTML."""

from __future__ import annotations

from pathlib import Path

import click

from sharp.cli import predict
from sharp.splat_viewer_gui import RunConfig, generate_viewer
from sharp.utils import io


DEFAULT_OUTPUT_DIR = Path("C:/Users/phill/Dropbox/Splats/outputs")


def _resolve_output_dir(input_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    if DEFAULT_OUTPUT_DIR.exists():
        return DEFAULT_OUTPUT_DIR
    return input_path.parent


def _print_line(msg: str) -> None:
    end = "" if msg.endswith("\n") else "\n"
    print(msg, end=end, flush=True)


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to the input image.",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory to write the PLY + HTML outputs (defaults to input folder or the SHARP output folder if it exists).",
)
@click.option(
    "--output-html",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional path for the output HTML file (defaults to <output-dir>/<stem>.html).",
)
@click.option(
    "-c",
    "--checkpoint-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional SHARP checkpoint (.pt).",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    show_default=True,
    help="Device to run SHARP prediction on. ['cpu', 'mps', 'cuda']",
)
@click.option(
    "--viewer-settings",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional splat-transform viewer settings JSON.",
)
@click.option("--unbundled", is_flag=True, help="Output unbundled viewer assets.")
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    show_default=True,
    help="Overwrite existing output files.",
)
@click.option(
    "--aa/--no-aa",
    "enable_aa",
    default=True,
    show_default=True,
    help="Enable AA by default in the viewer.",
)
@click.option(
    "--msaa/--no-msaa",
    "enable_msaa",
    default=True,
    show_default=True,
    help="Enable MSAA by default in the viewer.",
)
@click.option(
    "--match-gsplat-camera/--no-match-gsplat-camera",
    default=True,
    show_default=True,
    help="Match the gsplat camera metadata in the viewer.",
)
@click.option(
    "--splat-gpu",
    type=str,
    default="default",
    show_default=True,
    help="splat-transform GPU selector (e.g. 'cpu' or '0').",
)
@click.option(
    "--iterations",
    type=int,
    default=18,
    show_default=True,
    help="splat-transform iteration count.",
)
@click.option(
    "--extra-args",
    type=str,
    default="",
    help="Extra arguments passed to splat-transform.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def photo_to_viewer_cli(
    input_path: Path,
    output_dir: Path | None,
    output_html: Path | None,
    checkpoint_path: Path | None,
    device: str,
    viewer_settings: Path | None,
    unbundled: bool,
    overwrite: bool,
    enable_aa: bool,
    enable_msaa: bool,
    match_gsplat_camera: bool,
    splat_gpu: str,
    iterations: int,
    extra_args: str,
    verbose: bool,
) -> None:
    """Run SHARP prediction then generate a self-contained splat viewer HTML."""
    supported = io.get_supported_image_extensions()
    if input_path.suffix not in supported:
        raise click.UsageError(f"Unsupported image extension: {input_path.suffix}")

    output_dir = _resolve_output_dir(input_path, output_dir)
    output_html = output_html or (output_dir / f"{input_path.stem}.html")
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_line(f"Predicting Gaussians PLY to {output_dir}...")
    predict.predict_cli.callback(
        input_path=input_path,
        output_path=output_dir,
        checkpoint_path=checkpoint_path,
        with_rendering=False,
        fps=60.0,
        video_ext="mp4",
        write_frames=False,
        frame_ext="png",
        jpeg_quality=92,
        codec=None,
        bitrate=None,
        macro_block_size=16,
        render_depth=True,
        duration_scale=8.0,
        trajectory_type=None,
        lookat_mode=None,
        max_disparity=None,
        max_zoom=None,
        distance_m=None,
        num_steps=None,
        num_repeats=None,
        trajectory_variants=False,
        trajectory_variants_count=5,
        output_prefix="",
        overwrite=overwrite,
        low_pass_filter_eps=0.0,
        cuda_device=0,
        progress=True,
        device=device,
        verbose=verbose,
    )

    ply_path = output_dir / f"{input_path.stem}.ply"
    if not ply_path.is_file():
        raise click.ClickException(f"Expected PLY was not created: {ply_path}")

    if output_html.exists() and not overwrite:
        raise click.ClickException(f"Output HTML already exists: {output_html}")

    _print_line(f"Generating viewer HTML at {output_html}...")
    gpu_value = None if splat_gpu.strip().lower() == "default" else splat_gpu.strip()
    cfg = RunConfig(
        input_path=ply_path,
        output_html=output_html,
        viewer_settings=viewer_settings,
        unbundled=unbundled,
        overwrite=overwrite,
        enable_aa=enable_aa,
        enable_msaa=enable_msaa,
        match_gsplat_camera=match_gsplat_camera,
        gpu=gpu_value,
        iterations=iterations,
        extra_args=extra_args,
    )
    generate_viewer(cfg, log_line=_print_line)
