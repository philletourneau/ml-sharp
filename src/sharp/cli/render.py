"""Contains `sharp render` CLI implementation.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
import random
from pathlib import Path

import click
import torch
import torch.utils.data

from sharp.utils import camera, gsplat, io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import Gaussians3D, SceneMetaData, load_ply

LOGGER = logging.getLogger(__name__)

_PROGRESS_UPDATE_INTERVAL_S = 2.0


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes:d}m{rem:02.0f}s"
    hours = int(minutes // 60)
    rem_m = minutes - hours * 60
    return f"{hours:d}h{rem_m:02d}m"


def _iter_with_progress(iterable, *, total: int, desc: str, enabled: bool):
    """Iterate with elapsed/ETA progress output."""
    if not enabled:
        yield from iterable
        return

    try:
        if os.environ.get("SHARP_PROGRESS_STYLE", "").lower() in {"text", "plain"}:
            raise RuntimeError("tqdm disabled via SHARP_PROGRESS_STYLE")
        from tqdm.auto import tqdm  # type: ignore[import-not-found]

        yield from tqdm(
            iterable,
            total=total,
            desc=desc,
            unit="frame",
            dynamic_ncols=True,
            file=sys.stderr,
        )
        return
    except Exception:
        pass

    start_t = time.perf_counter()
    last_print_t = start_t
    for index, item in enumerate(iterable, start=1):
        now_t = time.perf_counter()
        should_print = (
            index == 1
            or (now_t - last_print_t) >= _PROGRESS_UPDATE_INTERVAL_S
            or index == total
        )
        if should_print:
            elapsed_s = now_t - start_t
            rate = index / elapsed_s if elapsed_s > 0 else 0.0
            remaining = max(0, total - index)
            eta_s = (remaining / rate) if rate > 0 else float("inf")
            click.echo(
                f"{desc}: {index}/{total} frames | elapsed {_format_duration(elapsed_s)} | "
                f"ETA {_format_duration(eta_s)}",
                err=True,
            )
            last_print_t = now_t
        yield item


def _resolve_output_path(path: Path, *, overwrite: bool) -> Path:
    """Resolve output path depending on overwrite policy."""
    if overwrite or not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for i in range(1, 10_000):
        candidate = path.with_name(f"{stem}_{i:03d}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an available output filename for {path}.")


def _make_trajectory_params(
    *,
    base: camera.TrajectoryParams,
    trajectory_type: camera.TrajetoryType | None,
    lookat_mode: camera.LookAtMode | None,
    max_disparity: float | None,
    max_zoom: float | None,
    distance_m: float | None,
    num_steps: int | None,
    num_repeats: int | None,
    duration_scale: float,
) -> camera.TrajectoryParams:
    if duration_scale <= 0:
        raise ValueError("duration_scale must be > 0.")

    params = base
    if trajectory_type is not None:
        params = dataclasses.replace(params, type=trajectory_type)
    if lookat_mode is not None:
        params = dataclasses.replace(params, lookat_mode=lookat_mode)
    if max_disparity is not None:
        params = dataclasses.replace(params, max_disparity=max_disparity)
    if max_zoom is not None:
        params = dataclasses.replace(params, max_zoom=max_zoom)
    if distance_m is not None:
        params = dataclasses.replace(params, distance_m=distance_m)
    if num_steps is not None:
        params = dataclasses.replace(params, num_steps=num_steps)
    else:
        scaled_steps = max(1, int(round(params.num_steps * duration_scale)))
        params = dataclasses.replace(params, num_steps=scaled_steps)
    if num_repeats is not None:
        params = dataclasses.replace(params, num_repeats=num_repeats)
    return params


@click.command()
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to the ply or a list of plys.",
    required=True,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    help="Path to save the rendered videos.",
    required=True,
)
@click.option(
    "--fps",
    type=float,
    default=60.0,
    show_default=True,
    help="Frames per second for the output videos.",
)
@click.option(
    "--video-ext",
    type=click.Choice(["mp4", "mov"], case_sensitive=False),
    default="mp4",
    show_default=True,
    help="Video container extension.",
)
@click.option(
    "--codec",
    type=str,
    default=None,
    help="FFmpeg codec name passed to imageio (optional).",
)
@click.option(
    "--bitrate",
    type=str,
    default=None,
    help="FFmpeg bitrate passed to imageio, e.g. '8M' (optional).",
)
@click.option(
    "--macro-block-size",
    type=int,
    default=16,
    show_default=True,
    help="FFmpeg macro block size for imageio (set 1 to disable resizing warnings).",
)
@click.option(
    "--depth/--no-depth",
    "render_depth",
    default=True,
    show_default=True,
    help="Whether to render a depth video alongside the color video.",
)
@click.option(
    "--duration-scale",
    type=float,
    default=8.0,
    show_default=True,
    help="Scale trajectory length by multiplying the number of frames.",
)
@click.option(
    "--trajectory-type",
    type=click.Choice(["swipe", "shake", "rotate", "rotate_forward"], case_sensitive=False),
    default=None,
    help="Camera trajectory type override.",
)
@click.option(
    "--lookat-mode",
    type=click.Choice(["point", "ahead"], case_sensitive=False),
    default=None,
    help="Look-at mode override.",
)
@click.option("--max-disparity", type=float, default=None, help="Override trajectory max disparity.")
@click.option("--max-zoom", type=float, default=None, help="Override trajectory max zoom.")
@click.option("--distance-m", type=float, default=None, help="Override trajectory camera distance in meters.")
@click.option("--num-steps", type=int, default=None, help="Override trajectory number of steps (frames per repeat).")
@click.option("--num-repeats", type=int, default=None, help="Override trajectory number of repeats.")
@click.option(
    "--trajectory-variants/--no-trajectory-variants",
    default=False,
    show_default=True,
    help="Render multiple variations of the trajectory (writes multiple output videos).",
)
@click.option(
    "--trajectory-variants-count",
    type=click.IntRange(1, 5),
    default=5,
    show_default=True,
    help="Number of trajectory variations to render (only used with --trajectory-variants).",
)
@click.option(
    "--output-prefix",
    type=str,
    default="",
    help="Prefix for output filenames (e.g. 'previz_').",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    show_default=True,
    help="Overwrite existing output files.",
)
@click.option(
    "--low-pass-filter-eps",
    type=float,
    default=0.0,
    show_default=True,
    help="Low-pass filter epsilon for gsplat rasterization.",
)
@click.option(
    "--cuda-device",
    type=int,
    default=0,
    show_default=True,
    help="CUDA device index to use.",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Show per-frame progress (elapsed and ETA).",
)
@click.option("-v", "--verbose", is_flag=True, help="Activate debug logs.")
def render_cli(
    input_path: Path,
    output_path: Path,
    fps: float,
    video_ext: str,
    codec: str | None,
    bitrate: str | None,
    macro_block_size: int,
    render_depth: bool,
    duration_scale: float,
    trajectory_type: str | None,
    lookat_mode: str | None,
    max_disparity: float | None,
    max_zoom: float | None,
    distance_m: float | None,
    num_steps: int | None,
    num_repeats: int | None,
    trajectory_variants: bool,
    trajectory_variants_count: int,
    output_prefix: str,
    overwrite: bool,
    low_pass_filter_eps: float,
    cuda_device: int,
    progress: bool,
    verbose: bool,
):
    """Render trajectory videos from Gaussians (.ply)."""
    logging_utils.configure(logging.DEBUG if verbose else logging.INFO)

    if not torch.cuda.is_available():
        LOGGER.error("Rendering a checkpoint requires CUDA.")
        exit(1)

    try:
        gsplat.require_gsplat_cuda_extension()
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        exit(1)

    output_path.mkdir(exist_ok=True, parents=True)

    base_params = camera.TrajectoryParams()
    variants_enabled = trajectory_variants and trajectory_variants_count > 1

    base_params = _make_trajectory_params(
        base=base_params,
        trajectory_type=trajectory_type.lower() if trajectory_type else None,
        lookat_mode=lookat_mode.lower() if lookat_mode else None,
        max_disparity=max_disparity,
        max_zoom=max_zoom,
        distance_m=distance_m,
        num_steps=num_steps,
        num_repeats=num_repeats,
        duration_scale=duration_scale,
    )

    if input_path.suffix == ".ply":
        scene_paths = [input_path]
    elif input_path.is_dir():
        scene_paths = list(input_path.glob("*.ply"))
    else:
        LOGGER.error("Input path must be either directory or single PLY file.")
        exit(1)

    for scene_path in scene_paths:
        LOGGER.info("Rendering %s", scene_path)
        gaussians, metadata = load_ply(scene_path)
        for variant_index, (params, suffix) in enumerate(
            _iter_trajectory_variants(
                base_params, enabled=variants_enabled, count=trajectory_variants_count
            )
        ):
            output_video = output_path / f"{output_prefix}{scene_path.stem}{suffix}.{video_ext}"
            output_video = _resolve_output_path(output_video, overwrite=overwrite)
            if variants_enabled:
                LOGGER.info("Trajectory variant %d -> %s", variant_index, output_video)
            render_gaussians(
                gaussians=gaussians,
                metadata=metadata,
                params=params,
                output_path=output_video,
                fps=fps,
                codec=codec,
                bitrate=bitrate,
                macro_block_size=macro_block_size,
                render_depth=render_depth,
                low_pass_filter_eps=low_pass_filter_eps,
                cuda_device=cuda_device,
                progress=progress,
            )


def render_gaussians(
    gaussians: Gaussians3D,
    metadata: SceneMetaData,
    output_path: Path,
    params: camera.TrajectoryParams | None = None,
    fps: float = 60.0,
    codec: str | None = None,
    bitrate: str | None = None,
    macro_block_size: int = 16,
    render_depth: bool = True,
    low_pass_filter_eps: float = 0.0,
    cuda_device: int = 0,
    progress: bool = True,
) -> None:
    """Render a single gaussian checkpoint file."""
    (width, height) = metadata.resolution_px
    f_px = metadata.focal_length_px

    if params is None:
        params = camera.TrajectoryParams()

    if not torch.cuda.is_available():
        raise RuntimeError("Rendering a checkpoint requires CUDA.")

    gsplat.require_gsplat_cuda_extension()

    device = torch.device(f"cuda:{cuda_device}")

    intrinsics = torch.tensor(
        [
            [f_px, 0, (width - 1) / 2., 0],
            [0, f_px, (height - 1) / 2., 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    camera_model = camera.create_camera_model(
        gaussians, intrinsics, resolution_px=metadata.resolution_px, lookat_mode=params.lookat_mode
    )

    trajectory = camera.create_eye_trajectory(
        gaussians, params, resolution_px=metadata.resolution_px, f_px=f_px
    )
    total_frames = len(trajectory)
    LOGGER.info(
        "Trajectory: %d frames @ %.1f fps (~%s)",
        total_frames,
        fps,
        _format_duration(total_frames / fps if fps > 0 else 0.0),
    )
    renderer = gsplat.GSplatRenderer(color_space=metadata.color_space, low_pass_filter_eps=low_pass_filter_eps)
    video_writer = io.VideoWriter(
        output_path,
        fps=fps,
        render_depth=render_depth,
        codec=codec,
        bitrate=bitrate,
        macro_block_size=macro_block_size,
    )
    gaussians_device = gaussians.to(device)

    for eye_position in _iter_with_progress(
        trajectory, total=total_frames, desc=str(output_path.name), enabled=progress
    ):
        camera_info = camera_model.compute(eye_position)
        rendering_output = renderer(
            gaussians_device,
            extrinsics=camera_info.extrinsics[None].to(device),
            intrinsics=camera_info.intrinsics[None].to(device),
            image_width=camera_info.width,
            image_height=camera_info.height,
        )
        color = (rendering_output.color[0].permute(1, 2, 0) * 255.0).to(dtype=torch.uint8)
        depth = rendering_output.depth[0]
        video_writer.add_frame(color, depth)
    video_writer.close()


def _iter_trajectory_variants(
    base_params: camera.TrajectoryParams,
    *,
    enabled: bool,
    count: int,
):
    """Yield (params, output_suffix) for trajectory variants."""
    if not enabled:
        yield base_params, ""
        return

    if count < 1:
        raise ValueError("count must be >= 1")

    # Use a fixed set of different motion types to ensure variants are visibly distinct.
    variant_types: list[camera.TrajetoryType] = [
        "rotate_forward",
        "rotate",
        "swipe",
        "shake",
        "rotate_forward",
    ]

    for variant_index in range(count):
        rng = random.Random(variant_index)
        disparity_scale = rng.uniform(0.6, 1.6)
        zoom_scale = rng.uniform(0.6, 1.6)
        distance_offset = rng.uniform(-0.2, 0.2)
        repeats = rng.choice([1, 2, 3])
        lookat_mode: camera.LookAtMode = rng.choice(["point", "ahead"])
        traj_type = variant_types[variant_index % len(variant_types)]
        params = dataclasses.replace(
            base_params,
            type=traj_type,
            lookat_mode=lookat_mode,
            max_disparity=base_params.max_disparity * disparity_scale,
            max_zoom=base_params.max_zoom * zoom_scale,
            distance_m=base_params.distance_m + distance_offset,
            num_repeats=repeats,
        )
        yield params, f"_v{variant_index:02d}"
